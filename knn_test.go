// Directed tests and a small example for the read-only 2D point
// k-nearest-neighbor query entry point (KNN).

package rtree

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"testing"
)

// insertPoint inserts a 2D point item into the tree.
func insertPoint(tr *RTreeG[int], x, y float64, data int) {
	tr.Insert([2]float64{x, y}, [2]float64{x, y}, data)
}

// scanKNN computes the k nearest items by brute-force full scan,
// ordering by (true Euclidean distance, stable insertion ID).
func scanKNN(tr *RTreeG[int], k int, point [2]float64) []KNNItem[int] {
	var all []KNNItem[int]
	tr.Scan(func(min, max [2]float64, data int) bool {
		dx, dy := min[0]-point[0], min[1]-point[1]
		all = append(all, KNNItem[int]{
			Min: min, Max: max, Data: data,
			Dist: math.Sqrt(dx*dx + dy*dy),
			// These tests always insert data values 0..n-1 in order,
			// so the data payload doubles as the stable insertion ID.
			ID: uint64(data),
		})
		return true
	})
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Dist != all[j].Dist {
			return all[i].Dist < all[j].Dist
		}
		return all[i].ID < all[j].ID
	})
	if len(all) > k {
		all = all[:k]
	}
	return all
}

func sameKNN(a, b []KNNItem[int]) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Data != b[i].Data || a[i].Dist != b[i].Dist ||
			a[i].ID != b[i].ID || a[i].Min != b[i].Min ||
			a[i].Max != b[i].Max {
			return false
		}
	}
	return true
}

// TestKNNFivePoints checks that k=2 over five points matches a
// brute-force full scan.
func TestKNNFivePoints(t *testing.T) {
	var tr RTreeG[int]
	pts := [][2]float64{
		{3, 4}, {1, 1}, {5, 0}, {0, -2}, {2, 2},
	}
	for i, p := range pts {
		insertPoint(&tr, p[0], p[1], i)
	}
	query := [2]float64{0, 0}
	got, err := tr.KNN(2, query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := scanKNN(&tr, 2, query)
	if !sameKNN(got, want) {
		t.Fatalf("knn mismatch:\n got: %+v\nwant: %+v", got, want)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
	// The two nearest points to (0,0) are {1,1} (dist sqrt2) and
	// {0,-2} (dist 2), reported with true Euclidean distances.
	if got[0].Data != 1 || got[0].Dist != math.Sqrt(2) {
		t.Fatalf("bad first result: %+v", got[0])
	}
	if got[1].Data != 3 || got[1].Dist != 2 {
		t.Fatalf("bad second result: %+v", got[1])
	}
}

// countNodes counts all nodes in the tree.
func countNodes[N numeric, T any](n *node[N, T]) int {
	if n == nil {
		return 0
	}
	if n.leaf() {
		return 1
	}
	count := 1
	for _, c := range n.children()[:n.count] {
		count += countNodes(c)
	}
	return count
}

// TestKNNPruning checks that nodes far from the query point are pruned
// (never expanded) while the results still match a full scan.
func TestKNNPruning(t *testing.T) {
	var tr RTreeG[int]
	// A tight cluster near the origin and a large far-away cluster,
	// with enough points to force branch nodes.
	for i := 0; i < 200; i++ {
		insertPoint(&tr, float64(i%10), float64(i/10), i)
	}
	for i := 0; i < 200; i++ {
		insertPoint(&tr, 10000+float64(i%10), 10000+float64(i/10), 200+i)
	}
	total := countNodes(tr.base.root)
	if total < 3 {
		t.Fatalf("expected a multi-node tree, got %d nodes", total)
	}
	got, visited := tr.base.knn(2, 0, 0)
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
	want := scanKNN(&tr, 2, [2]float64{0, 0})
	if !sameKNN(got, want) {
		t.Fatalf("knn mismatch:\n got: %+v\nwant: %+v", got, want)
	}
	if visited >= total {
		t.Fatalf("expected pruning: visited %d of %d nodes", visited, total)
	}
	for _, item := range got {
		if item.Data >= 200 {
			t.Fatalf("far-away item in results: %+v", item)
		}
	}
}

// TestKNNTies checks that items with equal distances are ordered by their
// stable insertion ID, and that the order is fixed across repeated
// queries.
func TestKNNTies(t *testing.T) {
	var tr RTreeG[int]
	// Four points all at distance 1 from the origin, inserted in a
	// non-symmetric order.
	pts := [][2]float64{{0, 1}, {-1, 0}, {1, 0}, {0, -1}}
	for i, p := range pts {
		insertPoint(&tr, p[0], p[1], i)
	}
	first, err := tr.KNN(4, [2]float64{0, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(first) != 4 {
		t.Fatalf("expected 4 results, got %d", len(first))
	}
	for i, item := range first {
		if item.Dist != 1 {
			t.Fatalf("expected distance 1, got %v", item.Dist)
		}
		if item.Data != i || item.ID != uint64(i) {
			t.Fatalf("tie not ordered by stable ID at %d: %+v", i, item)
		}
	}
	// k=2 must select the two tied items with the smallest stable IDs.
	top2, err := tr.KNN(2, [2]float64{0, 0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(top2) != 2 || top2[0].Data != 0 || top2[1].Data != 1 {
		t.Fatalf("bad tie cut: %+v", top2)
	}
	// Repeated queries must return the exact same order.
	for i := 0; i < 3; i++ {
		again, err := tr.KNN(4, [2]float64{0, 0})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !sameKNN(first, again) {
			t.Fatalf("tie order not fixed:\nfirst: %+v\nagain: %+v",
				first, again)
		}
	}
}

// TestKNNRejectsNaN checks that a NaN query point is rejected and that
// the rejected query leaves the tree unchanged.
func TestKNNRejectsNaN(t *testing.T) {
	var tr RTreeG[int]
	pts := [][2]float64{{1, 2}, {3, 4}, {5, 6}}
	for i, p := range pts {
		insertPoint(&tr, p[0], p[1], i)
	}
	min, max := tr.Bounds()
	before := scanKNN(&tr, len(pts), [2]float64{0, 0})

	nan := math.NaN()
	for _, q := range [][2]float64{{nan, 0}, {0, nan}, {nan, nan}} {
		items, err := tr.KNN(2, q)
		if !errors.Is(err, ErrInvalidQueryPoint) {
			t.Fatalf("expected ErrInvalidQueryPoint, got %v", err)
		}
		if items != nil {
			t.Fatalf("expected nil results for NaN query, got %+v", items)
		}
	}

	// The tree must be unchanged.
	if tr.Len() != len(pts) {
		t.Fatalf("tree length changed: %d", tr.Len())
	}
	min2, max2 := tr.Bounds()
	if min != min2 || max != max2 {
		t.Fatalf("tree bounds changed: %v %v -> %v %v", min, max,
			min2, max2)
	}
	after := scanKNN(&tr, len(pts), [2]float64{0, 0})
	if !sameKNN(before, after) {
		t.Fatalf("tree content changed:\nbefore: %+v\nafter: %+v",
			before, after)
	}
	// The original query interfaces still work.
	var scanned int
	tr.Search([2]float64{0, 0}, [2]float64{4, 4},
		func(min, max [2]float64, data int) bool {
			scanned++
			return true
		})
	if scanned != 2 {
		t.Fatalf("Search returned %d items, expected 2", scanned)
	}
}

// TestKNNEdgeCases covers empty trees and non-positive k.
func TestKNNEdgeCases(t *testing.T) {
	var tr RTreeG[int]
	items, err := tr.KNN(3, [2]float64{0, 0})
	if err != nil || len(items) != 0 {
		t.Fatalf("empty tree: items=%v err=%v", items, err)
	}
	insertPoint(&tr, 1, 1, 7)
	items, err = tr.KNN(0, [2]float64{0, 0})
	if err != nil || len(items) != 0 {
		t.Fatalf("k=0: items=%v err=%v", items, err)
	}
	items, err = tr.KNN(5, [2]float64{0, 0})
	if err != nil || len(items) != 1 || items[0].Data != 7 {
		t.Fatalf("k>len: items=%v err=%v", items, err)
	}
}

// ExampleRTreeG_KNN is a small point-set example of the read-only
// k-nearest-neighbor query.
func ExampleRTreeG_KNN() {
	var tr RTreeG[string]
	cities := []struct {
		name string
		x, y float64
	}{
		{"London", 51.5, -0.12},
		{"Paris", 48.85, 2.35},
		{"Berlin", 52.52, 13.40},
		{"Madrid", 40.41, -3.70},
		{"Rome", 41.90, 12.50},
	}
	for _, c := range cities {
		tr.Insert([2]float64{c.x, c.y}, [2]float64{c.x, c.y}, c.name)
	}
	// Find the 2 cities nearest to Brussels (50.85, 4.35).
	items, err := tr.KNN(2, [2]float64{50.85, 4.35})
	if err != nil {
		panic(err)
	}
	for _, item := range items {
		fmt.Printf("%s %.2f\n", item.Data, item.Dist)
	}
	// Output:
	// Paris 2.83
	// London 4.52
}
