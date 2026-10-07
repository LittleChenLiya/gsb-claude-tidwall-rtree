// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package rtree

// SnapshotItemG is an independent copy of a single entry returned by
// SearchSnapshot. Min and Max are value copies of the entry rectangle stored
// in the tree, so mutating them never affects the index. Data is the original
// value held by the tree; it is returned as-is and is not deep copied.
type SnapshotItemG[N numeric, T any] struct {
	Min  [2]N
	Max  [2]N
	Data T
}

// SearchSnapshot returns a point-in-time snapshot of every item that
// intersects the provided rectangle. Each result holds a value copy of the
// entry's current bounds together with its original data value (data is not
// deep copied). The same intersection rule as Search is used, so zero-area
// rectangles (points or line segments) match anything that touches their
// boundary. Mutating the returned Min/Max values cannot corrupt the tree, and
// later inserts, deletes, or replacements never alter a previously returned
// snapshot.
func (tr *RTreeGN[N, T]) SearchSnapshot(min, max [2]N) []SnapshotItemG[N, T] {
	var items []SnapshotItemG[N, T]
	tr.Search(min, max, func(imin, imax [2]N, data T) bool {
		items = append(items, SnapshotItemG[N, T]{
			Min:  imin,
			Max:  imax,
			Data: data,
		})
		return true
	})
	return items
}

// SearchSnapshot returns a point-in-time snapshot of every item that
// intersects the provided rectangle. Each result holds a value copy of the
// entry's current bounds together with its original data value (data is not
// deep copied). Mutating the returned bounds cannot corrupt the tree, and
// later deletes never alter the snapshot.
func (tr *RTreeG[T]) SearchSnapshot(
	min, max [2]float64,
) []SnapshotItemG[float64, T] {
	return tr.base.SearchSnapshot(min, max)
}

// SnapshotItem is an independent copy of a single entry returned by
// RTree.SearchSnapshot. Min and Max are value copies of the entry rectangle,
// while Data is the original value and is not deep copied.
type SnapshotItem = SnapshotItemG[float64, any]

// SearchSnapshot returns a point-in-time snapshot of every item that
// intersects the provided rectangle. Each result holds a value copy of the
// entry's current bounds together with its original data value (data is not
// deep copied). Mutating the returned bounds cannot corrupt the tree, and
// later deletes never alter the snapshot.
func (tr *RTree) SearchSnapshot(
	min, max [2]float64,
) []SnapshotItem {
	return tr.base.SearchSnapshot(min, max)
}
