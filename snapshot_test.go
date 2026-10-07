// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package rtree

import (
	"fmt"
	"sort"
	"testing"
)

// snapItem is a test-only re-description of a snapshot entry.
type snapItem[N numeric] struct {
	data   string
	min    [2]N
	max    [2]N
}

func collectSnapshotG(tr *RTreeG[string], min, max [2]float64) []snapItem[float64] {
	items := tr.SearchSnapshot(min, max)
	out := make([]snapItem[float64], len(items))
	for i, it := range items {
		out[i] = snapItem[float64]{it.Data, it.Min, it.Max}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].data < out[j].data })
	return out
}

func collectSearchG(tr *RTreeG[string], min, max [2]float64) []snapItem[float64] {
	var out []snapItem[float64]
	tr.Search(min, max, func(min, max [2]float64, data string) bool {
		out = append(out, snapItem[float64]{data, min, max})
		return true
	})
	sort.Slice(out, func(i, j int) bool { return out[i].data < out[j].data })
	return out
}

// TestSearchSnapshot covers the small-fixture acceptance scenario:
// query two small rectangles, mutate the returned boundary copies, delete an
// entry, repeat the query, and verify snapshots / live tree / zero-area rules.
func TestSearchSnapshot(t *testing.T) {
	var tr RTreeG[string]
	// two small rectangles plus a zero-area point sitting on the corner of "a"
	tr.Insert([2]float64{0, 0}, [2]float64{10, 10}, "a")
	tr.Insert([2]float64{20, 20}, [2]float64{30, 30}, "b")
	tr.Insert([2]float64{10, 10}, [2]float64{10, 10}, "p")

	if tr.Len() != 3 {
		t.Fatalf("expected len 3, got %d", tr.Len())
	}

	// 1) Snapshot entry must agree with the original Search interface.
	snap := tr.SearchSnapshot([2]float64{0, 0}, [2]float64{15, 15})
	live := collectSearchG(&tr, [2]float64{0, 0}, [2]float64{15, 15})
	got := make([]snapItem[float64], len(snap))
	for i, it := range snap {
		got[i] = snapItem[float64]{it.Data, it.Min, it.Max}
	}
	sort.Slice(got, func(i, j int) bool { return got[i].data < got[j].data })
	want := []snapItem[float64]{
		{"a", [2]float64{0, 0}, [2]float64{10, 10}},
		{"p", [2]float64{10, 10}, [2]float64{10, 10}},
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("snapshot mismatch:\n got %v\nwant %v", got, want)
	}
	if fmt.Sprint(got) != fmt.Sprint(live) {
		t.Fatalf("snapshot %v does not match Search %v", got, live)
	}
	// second small window only sees "b"
	if b := collectSnapshotG(&tr, [2]float64{20, 20}, [2]float64{35, 35}); len(b) != 1 || b[0].data != "b" {
		t.Fatalf("expected only b, got %v", b)
	}

	// 2) Zero-area rectangles follow the original intersection rule: a point
	// query touching a corner/edge still intersects.
	touch := collectSnapshotG(&tr, [2]float64{10, 10}, [2]float64{10, 10})
	if s := dataNames(touch); s != "[a p]" {
		t.Fatalf("zero-area boundary touch expected [a p], got %s", s)
	}
	miss := collectSnapshotG(&tr, [2]float64{11, 11}, [2]float64{11, 11})
	if len(miss) != 0 {
		t.Fatalf("point just outside expected no hits, got %v", miss)
	}

	// 3) Mutate every boundary copy in the snapshot; the tree index must not
	// move at all.
	for i := range snap {
		snap[i].Min = [2]float64{900, 900}
		snap[i].Max = [2]float64{999, 999}
	}
	if tr.Len() != 3 {
		t.Fatalf("len changed after snapshot mutation: %d", tr.Len())
	}
	bmin, bmax := tr.Bounds()
	if bmin != ([2]float64{0, 0}) || bmax != ([2]float64{30, 30}) {
		t.Fatalf("tree bounds corrupted after snapshot mutation: %v %v", bmin, bmax)
	}
	liveAfter := collectSearchG(&tr, [2]float64{0, 0}, [2]float64{15, 15})
	if fmt.Sprint(liveAfter) != fmt.Sprint(want) {
		t.Fatalf("live tree rects changed after snapshot mutation: %v", liveAfter)
	}
	// a scan of the whole tree must still expose all original rectangles
	all := collectSearchG(&tr, [2]float64{-1, -1}, [2]float64{100, 100})
	if len(all) != 3 {
		t.Fatalf("expected 3 items after mutation, got %v", all)
	}

	// 4) Delete an entry and repeat the query. The old snapshot is immutable:
	// it still contains the deleted entry with its mutated copies.
	tr.Delete([2]float64{0, 0}, [2]float64{10, 10}, "a")
	if tr.Len() != 2 {
		t.Fatalf("expected len 2 after delete, got %d", tr.Len())
	}
	requery := collectSnapshotG(&tr, [2]float64{0, 0}, [2]float64{15, 15})
	if s := dataNames(requery); s != "[p]" {
		t.Fatalf("post-delete query expected [p], got %s", s)
	}
	if len(snap) != 2 || snap[0].Data != "a" || snap[0].Min != [2]float64{900, 900} ||
		snap[1].Data != "p" || snap[1].Max != [2]float64{999, 999} {
		t.Fatalf("old snapshot altered by delete: %+v", snap)
	}
	// deleted entry's zero-area neighbor still matches a boundary touch
	touch2 := collectSnapshotG(&tr, [2]float64{10, 10}, [2]float64{10, 10})
	if s := dataNames(touch2); s != "[p]" {
		t.Fatalf("post-delete touch expected [p], got %s", s)
	}
}

func dataNames[N numeric](items []snapItem[N]) string {
	names := make([]string, len(items))
	for i, it := range items {
		names[i] = it.data
	}
	sort.Strings(names)
	return fmt.Sprint(names)
}

// TestSearchSnapshotDataNotDeepCopied verifies that only the rectangle is
// copied: the original value is shared (no deep copy).
func TestSearchSnapshotDataNotDeepCopied(t *testing.T) {
	type payload struct {
		tag string
		n   int
	}
	var tr RTreeG[*payload]
	v := &payload{tag: "x", n: 1}
	tr.Insert([2]float64{0, 0}, [2]float64{1, 1}, v)

	snap := tr.SearchSnapshot([2]float64{0, 0}, [2]float64{1, 1})
	if len(snap) != 1 || snap[0].Data != v {
		t.Fatalf("expected shared pointer value, got %+v", snap)
	}
	// mutating through the snapshot value is visible to a live query
	snap[0].Data.n = 42
	seen := false
	tr.Search([2]float64{0, 0}, [2]float64{1, 1},
		func(min, max [2]float64, data *payload) bool {
			seen = data.n == 42
			return true
		},
	)
	if !seen {
		t.Fatal("snapshot data was deep-copied; expected shared value")
	}
}

// TestSearchSnapshotNumeric exercises the RTreeGN entry point with integer
// coordinates using enough items (65 > maxEntries) to force a branch split,
// plus a zero-area item, proving mutating snapshot rects survives in a
// branched tree too.
func TestSearchSnapshotNumeric(t *testing.T) {
	var tr RTreeGN[int, int]
	const n = 65
	for i := 0; i < n; i++ {
		tr.Insert([2]int{i, 0}, [2]int{i, 0}, i) // zero-area points
	}
	snap := tr.SearchSnapshot([2]int{0, 0}, [2]int{100, 0})
	if len(snap) != n {
		t.Fatalf("expected %d hits, got %d", n, len(snap))
	}
	for i := range snap {
		snap[i].Min = [2]int{-1, -1}
		snap[i].Max = [2]int{-1, -1}
	}
	if tr.Len() != n {
		t.Fatalf("len changed after snapshot mutation: %d", tr.Len())
	}
	min, max := tr.Bounds()
	if min != ([2]int{0, 0}) || max != ([2]int{n - 1, 0}) {
		t.Fatalf("bounds corrupted: %v %v", min, max)
	}
	count := 0
	tr.Search([2]int{0, 0}, [2]int{100, 0},
		func(min, max [2]int, data int) bool {
			if min != [2]int{data, 0} || max != [2]int{data, 0} {
				t.Fatalf("tree rect corrupted for %d: %v %v", data, min, max)
			}
			count++
			return true
		},
	)
	if count != n {
		t.Fatalf("expected %d live hits, got %d", n, count)
	}
	// delete through one half of the tree; the old snapshot keeps all entries
	for i := 0; i < n; i++ {
		tr.Delete([2]int{i, 0}, [2]int{i, 0}, i)
	}
	if tr.Len() != 0 {
		t.Fatalf("expected empty tree, len=%d", tr.Len())
	}
	if len(snap) != n {
		t.Fatalf("old snapshot shrank after deletes: %d", len(snap))
	}
}
