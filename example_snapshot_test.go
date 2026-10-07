// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package rtree

import (
	"fmt"
	"sort"
	"strings"
)

// ExampleRTree_SearchSnapshot is a short regression for the default RTree
// interface: query two small rectangles, mutate the returned boundary copies,
// delete an entry, and repeat the query while checking snapshots, live tree
// matches, and zero-area boundary behavior.
func ExampleRTree_SearchSnapshot() {
	var tr RTree
	tr.Insert([2]float64{0, 0}, [2]float64{10, 10}, "a")
	tr.Insert([2]float64{20, 20}, [2]float64{30, 30}, "b")
	tr.Insert([2]float64{10, 10}, [2]float64{10, 10}, "p") // zero-area point

	printSnap := func(tag string, items []Item[float64, any]) {
		sort.Slice(items, func(i, j int) bool {
			return items[i].Data.(string) < items[j].Data.(string)
		})
		for _, it := range items {
			fmt.Printf("%s %s %v %v\n", tag, it.Data, it.Min, it.Max)
		}
	}

	// query window: small rectangle a plus zero-area point p on its corner
	snap := tr.SearchSnapshot([2]float64{0, 0}, [2]float64{15, 15})
	printSnap("q1", snap)

	// zero-area query window touching a boundary still intersects
	printSnap("touch",
		tr.SearchSnapshot([2]float64{10, 10}, [2]float64{10, 10}))

	// mutate every snapshot boundary copy; the live tree index stays intact
	for i := range snap {
		snap[i].Min = [2]float64{900, 900}
		snap[i].Max = [2]float64{999, 999}
	}
	var live []string
	tr.Search([2]float64{0, 0}, [2]float64{15, 15},
		func(min, max [2]float64, data any) bool {
			live = append(live, fmt.Sprintf("%s%v%v", data, min, max))
			return true
		},
	)
	sort.Strings(live)
	fmt.Println("live:", strings.Join(live, " "))

	// delete a and repeat the query; the old snapshot is unchanged
	tr.Delete([2]float64{0, 0}, [2]float64{10, 10}, "a")
	printSnap("q2", tr.SearchSnapshot([2]float64{0, 0}, [2]float64{15, 15}))
	printSnap("old", snap)
	// Output:
	// q1 a [0 0] [10 10]
	// q1 p [10 10] [10 10]
	// touch a [0 0] [10 10]
	// touch p [10 10] [10 10]
	// live: a[0 0][10 10] p[10 10][10 10]
	// q2 p [10 10] [10 10]
	// old a [900 900] [999 999]
	// old p [900 900] [999 999]
}
