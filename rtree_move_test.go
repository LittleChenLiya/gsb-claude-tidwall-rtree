package rtree

import (
	"sort"
	"testing"
)

// moveItem is a small fixture item carrying a stable ID.
type moveItem struct {
	id  int
	tag string
}

func moveItemID(it moveItem) any { return it.id }

// searchIDs collects the IDs of all items intersecting the given box.
func searchIDs(tr *RTreeG[moveItem], min, max [2]float64) []int {
	var ids []int
	tr.Search(min, max, func(_, _ [2]float64, it moveItem) bool {
		ids = append(ids, it.id)
		return true
	})
	sort.Ints(ids)
	return ids
}

func idsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Two entries share the exact same box but have different IDs.
// Moving one must leave the other in place.
func TestMoveByIDSameBoxTwoEntries(t *testing.T) {
	tr := new(RTreeG[moveItem])
	boxAmin := [2]float64{1, 1}
	boxAmax := [2]float64{2, 2}
	tr.Insert(boxAmin, boxAmax, moveItem{id: 1, tag: "first"})
	tr.Insert(boxAmin, boxAmax, moveItem{id: 2, tag: "second"})

	boxBmin := [2]float64{10, 10}
	boxBmax := [2]float64{11, 11}
	if !tr.MoveByID(boxAmin, boxAmax, boxBmin, boxBmax, 1, moveItemID) {
		t.Fatal("expected move to succeed")
	}
	if tr.Len() != 2 {
		t.Fatalf("expected len 2, got %d", tr.Len())
	}
	// The old box must only contain the untouched entry (id 2).
	if ids := searchIDs(tr, boxAmin, boxAmax); !idsEqual(ids, []int{2}) {
		t.Fatalf("old box: expected [2], got %v", ids)
	}
	// The new box must contain the moved entry (id 1).
	if ids := searchIDs(tr, boxBmin, boxBmax); !idsEqual(ids, []int{1}) {
		t.Fatalf("new box: expected [1], got %v", ids)
	}
	// The moved entry keeps its data.
	tr.Search(boxBmin, boxBmax, func(_, _ [2]float64, it moveItem) bool {
		if it.tag != "first" {
			t.Fatalf("expected tag 'first', got %q", it.tag)
		}
		return true
	})
}

// An invalid new box (min > max) must be rejected and the old entry
// must remain queryable at its original box.
func TestMoveByIDInvalidBoxRejected(t *testing.T) {
	tr := new(RTreeG[moveItem])
	boxAmin := [2]float64{1, 1}
	boxAmax := [2]float64{2, 2}
	tr.Insert(boxAmin, boxAmax, moveItem{id: 1, tag: "first"})
	tr.Insert(boxAmin, boxAmax, moveItem{id: 2, tag: "second"})

	// min > max on the x axis
	badMin := [2]float64{11, 10}
	badMax := [2]float64{10, 11}
	if tr.MoveByID(boxAmin, boxAmax, badMin, badMax, 1, moveItemID) {
		t.Fatal("expected move with invalid box to be rejected")
	}
	if tr.Len() != 2 {
		t.Fatalf("expected len 2, got %d", tr.Len())
	}
	// Both entries must still be at the old box.
	if ids := searchIDs(tr, boxAmin, boxAmax); !idsEqual(ids, []int{1, 2}) {
		t.Fatalf("old box: expected [1 2], got %v", ids)
	}
	// Nothing may appear at the rejected box.
	if ids := searchIDs(tr, badMax, badMin); len(ids) != 0 {
		t.Fatalf("rejected box: expected no items, got %v", ids)
	}
}

// Moving an ID that does not exist must change nothing.
func TestMoveByIDMissingID(t *testing.T) {
	tr := new(RTreeG[moveItem])
	boxAmin := [2]float64{1, 1}
	boxAmax := [2]float64{2, 2}
	tr.Insert(boxAmin, boxAmax, moveItem{id: 1, tag: "first"})
	tr.Insert(boxAmin, boxAmax, moveItem{id: 2, tag: "second"})

	boxBmin := [2]float64{10, 10}
	boxBmax := [2]float64{11, 11}
	if tr.MoveByID(boxAmin, boxAmax, boxBmin, boxBmax, 99, moveItemID) {
		t.Fatal("expected move of unknown ID to fail")
	}
	if tr.Len() != 2 {
		t.Fatalf("expected len 2, got %d", tr.Len())
	}
	if ids := searchIDs(tr, boxAmin, boxAmax); !idsEqual(ids, []int{1, 2}) {
		t.Fatalf("old box: expected [1 2], got %v", ids)
	}
	if ids := searchIDs(tr, boxBmin, boxBmax); len(ids) != 0 {
		t.Fatalf("new box: expected no items, got %v", ids)
	}
}
