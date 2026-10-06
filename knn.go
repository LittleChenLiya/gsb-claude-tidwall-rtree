// Copyright 2021 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package rtree

import "math"

// KNNItem is a single result entry returned by KNN.
// Dist is the true Euclidean distance from the query point to the item
// point. It is never the point-to-rectangle lower bound that is used
// internally for pruning.
type KNNItem[N numeric, T any] struct {
	Min, Max [2]N
	Data     T
	Dist     float64
}

// knnStats optionally reports internal counters for a KNN query.
type knnStats struct {
	nodesExpanded int // number of tree nodes popped and expanded
}

// KNN performs a read-only k-nearest-neighbor query for two-dimensional
// point data. It returns up to k items whose points (min == max) are
// closest to the query point, ordered by ascending true Euclidean
// distance. Ties on distance are broken by the stable ID provided by
// idOf (smaller ID first). If idOf is nil, all IDs are treated as zero.
//
// The search prunes tree nodes using the Euclidean lower-bound distance
// from the query point to each node rectangle, but the final ranking and
// the returned Dist values always use the true point-to-point distance of
// each entry, never the rectangle bound.
//
// The query is rejected (nil is returned) when k <= 0, the tree is empty,
// or the query point contains NaN or infinite coordinates. Entries or
// nodes that produce a NaN distance are skipped. The tree is never
// modified by this operation.
func (tr *RTreeGN[N, T]) KNN(point [2]N, k int, idOf func(data T) uint64,
) []KNNItem[N, T] {
	return tr.knn(point, k, idOf, nil)
}

func (tr *RTreeGN[N, T]) knn(point [2]N, k int, idOf func(data T) uint64,
	stats *knnStats,
) []KNNItem[N, T] {
	if k <= 0 || tr.root == nil {
		return nil
	}
	px, py := float64(point[0]), float64(point[1])
	if math.IsNaN(px) || math.IsNaN(py) ||
		math.IsInf(px, 0) || math.IsInf(py, 0) {
		// reject invalid query point; tree is left unchanged
		return nil
	}

	var pq knnHeap[N, T]
	pq.push(knnQItem[N, T]{dist: 0, node: tr.root})

	var res []knnQItem[N, T] // current top-k candidates
	var worst knnQItem[N, T] // worst (largest dist, then largest id) in res
	haveWorst := false

	for {
		it, ok := pq.pop()
		if !ok {
			break
		}
		if haveWorst && it.dist > worst.dist {
			// Everything left in the queue is farther away than the
			// current k-th best true distance. Prune the rest.
			break
		}
		if it.node != nil {
			if stats != nil {
				stats.nodesExpanded++
			}
			n := it.node
			rects := n.rects[:n.count]
			if n.leaf() {
				items := n.items()[:n.count]
				for i := 0; i < len(items); i++ {
					d := knnPointDist(px, py, &rects[i])
					if math.IsNaN(d) {
						// reject invalid distance
						continue
					}
					if haveWorst && d > worst.dist {
						continue
					}
					var id uint64
					if idOf != nil {
						id = idOf(items[i])
					}
					pq.push(knnQItem[N, T]{
						dist: d, id: id, rect: rects[i],
						data: items[i], item: true,
					})
				}
			} else {
				children := n.children()[:n.count]
				for i := 0; i < len(children); i++ {
					d := knnPointRectDist(px, py, &rects[i])
					if math.IsNaN(d) {
						// reject invalid distance
						continue
					}
					if haveWorst && d > worst.dist {
						// node is too far away to ever make the
						// top-k; prune it.
						continue
					}
					pq.push(knnQItem[N, T]{
						dist: d, rect: rects[i],
						node: children[i],
					})
				}
			}
			continue
		}
		// it's an item with a true distance
		if len(res) < k {
			res = append(res, it)
		} else if it.dist < worst.dist ||
			(it.dist == worst.dist && it.id < worst.id) {
			// replace the worst candidate
			for i := range res {
				if res[i].dist == worst.dist && res[i].id == worst.id {
					res[i] = it
					break
				}
			}
		} else {
			continue
		}
		// recompute the worst candidate
		worst = res[0]
		for i := 1; i < len(res); i++ {
			if res[i].dist > worst.dist ||
				(res[i].dist == worst.dist && res[i].id > worst.id) {
				worst = res[i]
			}
		}
		haveWorst = len(res) == k
	}

	// sort final results by (dist, id)
	for i := 1; i < len(res); i++ {
		for j := i; j > 0; j-- {
			a, b := res[j-1], res[j]
			if a.dist > b.dist || (a.dist == b.dist && a.id > b.id) {
				res[j-1], res[j] = b, a
			} else {
				break
			}
		}
	}
	out := make([]KNNItem[N, T], len(res))
	for i, it := range res {
		out[i] = KNNItem[N, T]{
			Min:  it.rect.min,
			Max:  it.rect.max,
			Data: it.data,
			Dist: it.dist,
		}
	}
	return out
}

// knnPointRectDist returns the Euclidean lower-bound distance from the
// point (px,py) to the rectangle r (zero when the point is inside r).
func knnPointRectDist[N numeric](px, py float64, r *rect[N]) float64 {
	var dx, dy float64
	if v := float64(r.min[0]) - px; v > 0 {
		dx = v
	} else if v := px - float64(r.max[0]); v > 0 {
		dx = v
	}
	if v := float64(r.min[1]) - py; v > 0 {
		dy = v
	} else if v := py - float64(r.max[1]); v > 0 {
		dy = v
	}
	return math.Hypot(dx, dy)
}

// knnPointDist returns the true Euclidean distance from the query point
// to the entry point. Entries are expected to be points (min == max).
func knnPointDist[N numeric](px, py float64, r *rect[N]) float64 {
	return math.Hypot(float64(r.min[0])-px, float64(r.min[1])-py)
}

type knnQItem[N numeric, T any] struct {
	dist float64     // node: lower-bound dist; item: true dist
	id   uint64      // stable ID for tie-breaking (items only)
	rect rect[N]     // item or node rect
	data T           // item data
	node *node[N, T] // node to expand (nil for items)
	item bool        // true when this is a leaf item
}

// knnHeap is a min-heap ordered by distance, then by stable ID for items.
type knnHeap[N numeric, T any] []knnQItem[N, T]

func knnLess[N numeric, T any](a, b *knnQItem[N, T]) bool {
	if a.dist != b.dist {
		return a.dist < b.dist
	}
	if a.item != b.item {
		// expand nodes before emitting items at equal distance
		return !a.item
	}
	if a.item && a.id != b.id {
		return a.id < b.id
	}
	return false
}

func (q *knnHeap[N, T]) push(it knnQItem[N, T]) {
	*q = append(*q, it)
	nodes := *q
	i := len(nodes) - 1
	parent := (i - 1) / 2
	for ; i != 0 && knnLess(&nodes[i], &nodes[parent]); parent = (i - 1) / 2 {
		nodes[parent], nodes[i] = nodes[i], nodes[parent]
		i = parent
	}
}

func (q *knnHeap[N, T]) pop() (knnQItem[N, T], bool) {
	nodes := *q
	if len(nodes) == 0 {
		return knnQItem[N, T]{}, false
	}
	var n knnQItem[N, T]
	n, nodes[0] = nodes[0], nodes[len(nodes)-1]
	nodes = nodes[:len(nodes)-1]
	*q = nodes
	i := 0
	for {
		smallest := i
		left := i*2 + 1
		right := i*2 + 2
		if left < len(nodes) && knnLess(&nodes[left], &nodes[smallest]) {
			smallest = left
		}
		if right < len(nodes) && knnLess(&nodes[right], &nodes[smallest]) {
			smallest = right
		}
		if smallest == i {
			break
		}
		nodes[smallest], nodes[i] = nodes[i], nodes[smallest]
		i = smallest
	}
	return n, true
}

// KNN performs a read-only k-nearest-neighbor query for two-dimensional
// point data. See RTreeGN.KNN for details.
func (tr *RTreeG[T]) KNN(point [2]float64, k int, idOf func(data T) uint64,
) []KNNItem[float64, T] {
	return tr.base.KNN(point, k, idOf)
}
