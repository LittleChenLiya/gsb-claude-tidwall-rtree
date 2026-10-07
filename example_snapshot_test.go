// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package rtree

import (
	"fmt"
	"sort"
)

// ExampleRTree_SearchSnapshot is a short regression for the default
// (non-generic) RTree interface snapshot entry point: snapshot a couple of
// small rectangles (including a zero-area point), mutate the returned bounds,
// delete the entries, and repeat the query.
func ExampleRTree_SearchSnapshot() {
	var tr RTree
	tr.Insert([2]float64{0, 0}, [2]float64{2, 2}, "A")
	tr.Insert([2]float64{3, 3}, [2]float64{5, 5}, "B")
	tr.Insert([2]float64{1, 1}, [2]float64{1, 1}, "P") // zero-area point

	dump := func(items []SnapshotItem) []string {
		names := make([]string, 0, len(items))
		for _, it := range items {
			names = append(names, it.Data.(string))
		}
		sort.Strings(names)
		return names
	}

	// snapshot the two small rects (and the point)
	snap := tr.SearchSnapshot([2]float64{0, 0}, [2]float64{5, 5})
	fmt.Println("snapshot:", dump(snap))

	// zero-area corner point still intersects A under the original rule
	touch := tr.SearchSnapshot([2]float64{2, 2}, [2]float64{2, 2})
	fmt.Println("corner-touch:", dump(touch))

	// corrupting the caller-owned rectangles cannot affect the tree index
	for i := range snap {
		snap[i].Min = [2]float64{-99, -99}
		snap[i].Max = [2]float64{-98, -98}
	}
	again := tr.SearchSnapshot([2]float64{0, 0}, [2]float64{5, 5})
	fmt.Println("tree-after-mutate:", dump(again))

	// delete the two rectangles and repeat the query
	tr.Delete([2]float64{0, 0}, [2]float64{2, 2}, "A")
	tr.Delete([2]float64{3, 3}, [2]float64{5, 5}, "B")
	after := tr.SearchSnapshot([2]float64{0, 0}, [2]float64{5, 5})
	fmt.Println("tree-after-delete:", dump(after))

	// the old snapshot is untouched, and data was not deep copied
	fmt.Println("old-snapshot-data:", snap[0].Data)
	// Output:
	// snapshot: [A B P]
	// corner-touch: [A]
	// tree-after-mutate: [A B P]
	// tree-after-delete: [P]
	// old-snapshot-data: A
}
