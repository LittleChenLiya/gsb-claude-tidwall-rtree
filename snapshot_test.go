// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package rtree

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

type snapPayload struct {
	name string
}

type snapEntry struct {
	name string
	min  [2]float64
	max  [2]float64
}

// snapEntries collects a snapshot and returns a deterministic, name-sorted
// summary so the results can be compared regardless of leaf ordering.
func snapEntries(items []SnapshotItemG[float64, *snapPayload]) []snapEntry {
	out := make([]snapEntry, 0, len(items))
	for _, it := range items {
		out = append(out, snapEntry{it.Data.name, it.Min, it.Max})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out
}

func formatSnap(entries []snapEntry) string {
	if len(entries) == 0 {
		return "<none>"
	}
	s := ""
	for i, e := range entries {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%s:{(%g,%g)-(%g,%g)}", e.name,
			e.min[0], e.min[1], e.max[0], e.max[1])
	}
	return s
}

// TestSearchSnapshot drives the acceptance scenario: query two small
// rectangles, mutate the returned bounds, delete the entries, re-query, and
// verify snapshot stability, tree/match consistency, zero-area boundary
// behavior, and shallow (non-deep) data copies. A small text report is
// written into a temp-dir fixture and echoed back for manual checking.
func TestSearchSnapshot(t *testing.T) {
	var report []string
	logf := func(format string, args ...any) {
		line := fmt.Sprintf(format, args...)
		t.Log(line)
		report = append(report, line)
	}

	var tr RTreeG[*snapPayload]
	pA := &snapPayload{name: "A"}
	pB := &snapPayload{name: "B"}
	pP := &snapPayload{name: "P"}
	tr.Insert([2]float64{0, 0}, [2]float64{2, 2}, pA) // small rect A
	tr.Insert([2]float64{3, 3}, [2]float64{5, 5}, pB) // small rect B
	tr.Insert([2]float64{1, 1}, [2]float64{1, 1}, pP) // zero-area point P

	full := [2]float64{0, 0}
	fullMax := [2]float64{5, 5}

	// 1) Query the two small rects (and the point) and take a snapshot.
	snap1 := tr.SearchSnapshot(full, fullMax)
	if got, want := len(snap1), 3; got != want {
		t.Fatalf("snapshot len: got %d, want %d", got, want)
	}
	logf("snapshot-1: %s", formatSnap(snapEntries(snap1)))

	// The data value is shared, not deep copied: same pointer as in tree.
	liveData := map[string]*snapPayload{}
	tr.Scan(func(min, max [2]float64, data *snapPayload) bool {
		liveData[data.name] = data
		return true
	})
	for _, it := range snap1 {
		if it.Data != liveData[it.Data.name] {
			t.Fatalf("data for %s was deep copied; expected shared value", it.Data.name)
		}
	}
	logf("data-shared: A/B/P data pointers identical to tree values")

	// 2) Zero-area queries follow the original intersects rule: boundary
	// contact counts as an intersection.
	corner := tr.SearchSnapshot([2]float64{2, 2}, [2]float64{2, 2})
	if got := formatSnap(snapEntries(corner)); got != "A:{(0,0)-(2,2)}" {
		t.Fatalf("zero-area corner match: %s", got)
	}
	logf("zero-area (2,2) corner touch: %s", formatSnap(snapEntries(corner)))

	gap := tr.SearchSnapshot([2]float64{2.5, 2.5}, [2]float64{2.5, 2.5})
	if len(gap) != 0 {
		t.Fatalf("zero-area gap point should match nothing, got %s",
			formatSnap(snapEntries(gap)))
	}
	logf("zero-area (2.5,2.5) in the gap: %s", formatSnap(snapEntries(gap)))

	pointSelf := tr.SearchSnapshot([2]float64{1, 1}, [2]float64{1, 1})
	if got := formatSnap(snapEntries(pointSelf)); got !=
		"A:{(0,0)-(2,2)} P:{(1,1)-(1,1)}" {
		t.Fatalf("zero-area point self match: %s", got)
	}
	logf("zero-area (1,1) on point entry P: %s",
		formatSnap(snapEntries(pointSelf)))

	// 3) Take a second snapshot, then smash the bounds of every snapshot
	// entry. The tree index must be unaffected.
	snap2 := tr.SearchSnapshot(full, fullMax)
	for i := range snap1 {
		snap1[i].Min = [2]float64{100, 100}
		snap1[i].Max = [2]float64{200, 200}
	}
	bmin, bmax := tr.Bounds()
	if bmin != ([2]float64{0, 0}) || bmax != ([2]float64{5, 5}) {
		t.Fatalf("tree bounds changed after snapshot mutation: %v-%v", bmin, bmax)
	}
	live := tr.SearchSnapshot(full, fullMax)
	if got, want := formatSnap(snapEntries(live)),
		"A:{(0,0)-(2,2)} B:{(3,3)-(5,5)} P:{(1,1)-(1,1)}"; got != want {
		t.Fatalf("live tree results after mutation:\n got %s\nwant %s", got, want)
	}
	logf("after-mutate tree bounds: (0,0)-(5,5)")
	logf("after-mutate tree search: %s", formatSnap(snapEntries(live)))
	logf("mutated snapshot-1 (caller-owned copy): %s",
		formatSnap(snapEntries(snap1)))

	// 4) Delete the two small rects and repeat the query.
	tr.Delete([2]float64{0, 0}, [2]float64{2, 2}, pA)
	tr.Delete([2]float64{3, 3}, [2]float64{5, 5}, pB)
	if got, want := tr.Len(), 1; got != want {
		t.Fatalf("tree len after deletes: got %d, want %d", got, want)
	}
	snap3 := tr.SearchSnapshot(full, fullMax)
	if got, want := formatSnap(snapEntries(snap3)), "P:{(1,1)-(1,1)}"; got != want {
		t.Fatalf("snapshot after deletes: got %s, want %s", got, want)
	}
	logf("after-delete tree len: 1")
	logf("after-delete snapshot-3: %s", formatSnap(snapEntries(snap3)))

	// 5) Old snapshots remain exactly as the caller left them; the later
	// deletes do not retroactively change them.
	if got, want := formatSnap(snapEntries(snap2)),
		"A:{(0,0)-(2,2)} B:{(3,3)-(5,5)} P:{(1,1)-(1,1)}"; got != want {
		t.Fatalf("snapshot-2 changed after delete: got %s, want %s", got, want)
	}
	if got, want := formatSnap(snapEntries(snap1)),
		"A:{(100,100)-(200,200)} B:{(100,100)-(200,200)} P:{(100,100)-(200,200)}"; got != want {
		t.Fatalf("snapshot-1 changed after delete: got %s, want %s", got, want)
	}
	logf("snapshot-2 still intact after deletes: %s", formatSnap(snapEntries(snap2)))
	logf("snapshot-1 still holds caller mutations: %s",
		formatSnap(snapEntries(snap1)))
	if a := lookupSnap(snap2, "A"); a.Data != liveData["A"] {
		t.Fatalf("snapshot data pointer changed after delete")
	}
	logf("deleted entries still reachable via old snapshot data: A=%q B=%q",
		lookupSnap(snap2, "A").Data.name, lookupSnap(snap2, "B").Data.name)

	// 6) Persist the small in-memory fixture report to a temp file and echo
	// it back as the acceptance artifact.
	dir := t.TempDir()
	path := filepath.Join(dir, "snapshot-report.txt")
	if err := os.WriteFile(path, []byte(joinLines(report)), 0o644); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(path); err != nil {
		t.Fatal(err)
	} else {
		logf("report-fixture: %s", path)
		t.Logf("report contents:\n%s", string(b))
	}
}

func lookupSnap(items []SnapshotItemG[float64, *snapPayload], name string,
) SnapshotItemG[float64, *snapPayload] {
	for _, it := range items {
		if it.Data.name == name {
			return it
		}
	}
	return SnapshotItemG[float64, *snapPayload]{}
}

func joinLines(lines []string) string {
	out := ""
	for _, l := range lines {
		out += l + "\n"
	}
	return out
}

// TestSearchSnapshotNumeric covers the generic RTreeGN core with integer
// coordinates and value-type data.
func TestSearchSnapshotNumeric(t *testing.T) {
	var tr RTreeGN[int, string]
	tr.Insert([2]int{0, 0}, [2]int{2, 2}, "A")
	tr.Insert([2]int{3, 3}, [2]int{5, 5}, "B")
	tr.Insert([2]int{2, 0}, [2]int{2, 0}, "edge") // zero-area edge point

	got := tr.SearchSnapshot([2]int{2, 0}, [2]int{2, 0})
	names := make([]string, 0, len(got))
	for _, it := range got {
		names = append(names, it.Data)
	}
	sort.Strings(names)
	want := []string{"A", "edge"}
	if len(names) != len(want) || names[0] != want[0] || names[1] != want[1] {
		t.Fatalf("numeric snapshot: got %v, want %v", names, want)
	}
	got[0].Min = [2]int{999, 999}
	tr.Delete([2]int{0, 0}, [2]int{2, 2}, "A")
	if tr.Len() != 2 {
		t.Fatalf("numeric tree len: got %d, want 2", tr.Len())
	}
	t.Logf("numeric zero-area edge snapshot: %v; len after delete: %d",
		names, tr.Len())
}
