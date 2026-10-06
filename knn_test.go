package rtree

import (
	"fmt"
	"math"
	"sort"
	"testing"
)

func knnIDOf(data int) uint64 { return uint64(data) }

// knnBruteForce computes the kNN answer by scanning every item.
func knnBruteForce(tr *RTreeG[int], point [2]float64, k int,
) []KNNItem[float64, int] {
	var all []KNNItem[float64, int]
	tr.Scan(func(min, max [2]float64, data int) bool {
		d := math.Hypot(min[0]-point[0], min[1]-point[1])
		all = append(all, KNNItem[float64, int]{
			Min: min, Max: max, Data: data, Dist: d,
		})
		return true
	})
	sort.Slice(all, func(i, j int) bool {
		if all[i].Dist != all[j].Dist {
			return all[i].Dist < all[j].Dist
		}
		return all[i].Data < all[j].Data
	})
	if len(all) > k {
		all = all[:k]
	}
	return all
}

func knnSame(a, b []KNNItem[float64, int]) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Data != b[i].Data || a[i].Dist != b[i].Dist ||
			a[i].Min != b[i].Min || a[i].Max != b[i].Max {
			return false
		}
	}
	return true
}

// Five points, k=2: the KNN answer must match a full scan.
func TestKNNFivePoints(t *testing.T) {
	var tr RTreeG[int]
	pts := [][2]float64{
		{3, 4}, {6, 8}, {0, 2}, {2, 0}, {9, 12},
	}
	for i, p := range pts {
		tr.Insert(p, p, i+1)
	}
	q := [2]float64{0, 0}
	got := tr.KNN(q, 2, knnIDOf)
	want := knnBruteForce(&tr, q, 2)
	if !knnSame(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 results, got %d", len(got))
	}
	// nearest two are ids 3 (0,2) and 4 (2,0), tied at dist 2,
	// ordered by stable ID.
	if got[0].Data != 3 || got[1].Data != 4 {
		t.Fatalf("unexpected order: %v", got)
	}
	if got[0].Dist != 2 || got[1].Dist != 2 {
		t.Fatalf("unexpected dists: %v %v", got[0].Dist, got[1].Dist)
	}
}

// k larger than the number of items returns all items, sorted.
func TestKNNAllItems(t *testing.T) {
	var tr RTreeG[int]
	pts := [][2]float64{{1, 1}, {2, 2}, {3, 3}}
	for i, p := range pts {
		tr.Insert(p, p, i+1)
	}
	got := tr.KNN([2]float64{0, 0}, 10, knnIDOf)
	if len(got) != 3 {
		t.Fatalf("expected 3 results, got %d", len(got))
	}
	for i := 1; i < len(got); i++ {
		if got[i].Dist < got[i-1].Dist {
			t.Fatal("results not sorted by distance")
		}
	}
}

// Nodes far from the query point must be pruned.
func TestKNNPruning(t *testing.T) {
	var tr RTreeG[int]
	id := 0
	// cluster A near the origin
	for i := 0; i < 100; i++ {
		p := [2]float64{float64(i%10) * 0.1, float64(i/10) * 0.1}
		id++
		tr.Insert(p, p, id)
	}
	// cluster B far away
	for i := 0; i < 100; i++ {
		p := [2]float64{10000 + float64(i%10)*0.1, 10000 + float64(i/10)*0.1}
		id++
		tr.Insert(p, p, id)
	}
	totalNodes := knnCountNodes(&tr.base.root, 0)
	if totalNodes < 3 {
		t.Fatalf("expected a multi-node tree, got %d nodes", totalNodes)
	}
	var stats knnStats
	q := [2]float64{0, 0}
	got := tr.base.knn(q, 5, knnIDOf, &stats)
	want := knnBruteForce(&tr, q, 5)
	if !knnSame(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if stats.nodesExpanded >= totalNodes {
		t.Fatalf("expected pruning: expanded %d of %d nodes",
			stats.nodesExpanded, totalNodes)
	}
	for _, it := range got {
		if it.Dist > 2 {
			t.Fatalf("far item in results: %+v", it)
		}
	}
}

func knnCountNodes(n **node[float64, int], depth int) int {
	if *n == nil {
		return 0
	}
	count := 1
	if !(*n).leaf() {
		for _, c := range (*n).children()[:(*n).count] {
			c := c
			count += knnCountNodes(&c, depth+1)
		}
	}
	return count
}

// Equal distances must be ordered by the stable ID.
func TestKNNTieStableID(t *testing.T) {
	var tr RTreeG[int]
	// all four points are exactly distance 1 from the origin
	tr.Insert([2]float64{1, 0}, [2]float64{1, 0}, 10)
	tr.Insert([2]float64{-1, 0}, [2]float64{-1, 0}, 3)
	tr.Insert([2]float64{0, 1}, [2]float64{0, 1}, 7)
	tr.Insert([2]float64{0, -1}, [2]float64{0, -1}, 1)
	got := tr.KNN([2]float64{0, 0}, 4, knnIDOf)
	if len(got) != 4 {
		t.Fatalf("expected 4 results, got %d", len(got))
	}
	wantIDs := []int{1, 3, 7, 10}
	for i, id := range wantIDs {
		if got[i].Data != id {
			t.Fatalf("position %d: got id %d, want %d", i, got[i].Data, id)
		}
		if got[i].Dist != 1 {
			t.Fatalf("position %d: got dist %v, want 1", i, got[i].Dist)
		}
	}
	// k=2 keeps the two smallest IDs
	got = tr.KNN([2]float64{0, 0}, 2, knnIDOf)
	if len(got) != 2 || got[0].Data != 1 || got[1].Data != 3 {
		t.Fatalf("unexpected tie result: %v", got)
	}
}

// A NaN query point is rejected and the tree is left unchanged.
func TestKNNNaNRejected(t *testing.T) {
	var tr RTreeG[int]
	pts := [][2]float64{{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}
	for i, p := range pts {
		tr.Insert(p, p, i+1)
	}
	snapshot := func() (int, [2]float64, [2]float64, []int) {
		var ids []int
		tr.Scan(func(min, max [2]float64, data int) bool {
			ids = append(ids, data)
			return true
		})
		bmin, bmax := tr.Bounds()
		return tr.Len(), bmin, bmax, ids
	}
	n0, bmin0, bmax0, ids0 := snapshot()

	bad := [][2]float64{
		{math.NaN(), 0},
		{0, math.NaN()},
		{math.NaN(), math.NaN()},
		{math.Inf(1), 0},
		{0, math.Inf(-1)},
	}
	for _, q := range bad {
		if got := tr.KNN(q, 2, knnIDOf); got != nil {
			t.Fatalf("query %v: expected nil, got %v", q, got)
		}
	}
	// k <= 0 and empty tree are also rejected
	if got := tr.KNN([2]float64{0, 0}, 0, knnIDOf); got != nil {
		t.Fatalf("expected nil for k=0, got %v", got)
	}
	var empty RTreeG[int]
	if got := empty.KNN([2]float64{0, 0}, 2, knnIDOf); got != nil {
		t.Fatalf("expected nil for empty tree, got %v", got)
	}

	n1, bmin1, bmax1, ids1 := snapshot()
	if n0 != n1 || bmin0 != bmin1 || bmax0 != bmax1 ||
		fmt.Sprint(ids0) != fmt.Sprint(ids1) {
		t.Fatal("tree changed after rejected queries")
	}
}

// ExampleRTreeG_KNN uses a small five-point set with k=2.
func ExampleRTreeG_KNN() {
	var tr RTreeG[int]
	tr.Insert([2]float64{3, 4}, [2]float64{3, 4}, 1)
	tr.Insert([2]float64{6, 8}, [2]float64{6, 8}, 2)
	tr.Insert([2]float64{0, 2}, [2]float64{0, 2}, 3)
	tr.Insert([2]float64{2, 0}, [2]float64{2, 0}, 4)
	tr.Insert([2]float64{9, 12}, [2]float64{9, 12}, 5)
	for _, it := range tr.KNN([2]float64{0, 0}, 2, func(d int) uint64 {
		return uint64(d)
	}) {
		fmt.Printf("id=%d dist=%g\n", it.Data, it.Dist)
	}
	// Output:
	// id=3 dist=2
	// id=4 dist=2
}
