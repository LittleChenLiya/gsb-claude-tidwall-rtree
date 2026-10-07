// Tests for the optional query-callback reentrancy protection.
//
// These tests use small in-memory fixtures plus a temporary-directory
// report fixture (TestQueryReentryReport) to demonstrate the behavior.

package rtree

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// makeProtectedTree builds a small tree with query protection enabled.
func makeProtectedTree() *RTreeG[int] {
	tr := new(RTreeG[int])
	for i := 0; i < 10; i++ {
		x := float64(i)
		tr.Insert([2]float64{x, x}, [2]float64{x, x}, i)
	}
	tr.SetQueryProtection(true)
	return tr
}

// Case 1: callback inserts into the SAME tree -> ErrQueryReentry,
// mutation rejected, protection released afterwards.
func TestQueryReentryInsertSameTree(t *testing.T) {
	tr := makeProtectedTree()
	if !tr.QueryProtection() {
		t.Fatal("expected protection to be enabled")
	}
	before := tr.Len()
	var visited int
	err := tr.SearchE([2]float64{0, 0}, [2]float64{9, 9},
		func(min, max [2]float64, data int) bool {
			visited++
			// attempt a reentrant mutation of the tree being queried
			tr.Insert([2]float64{100, 100}, [2]float64{100, 100}, 100)
			return true
		})
	if !errors.Is(err, ErrQueryReentry) {
		t.Fatalf("expected ErrQueryReentry, got %v", err)
	}
	if tr.Len() != before {
		t.Fatalf("reentrant insert was not rejected: len=%d, want %d",
			tr.Len(), before)
	}
	if visited == 0 {
		t.Fatal("callback was never invoked")
	}
	t.Logf("insert-during-search: err=%v visited=%d len=%d (unchanged)",
		err, visited, tr.Len())

	// protection must be released: mutating after the query works
	tr.Insert([2]float64{100, 100}, [2]float64{100, 100}, 100)
	if tr.Len() != before+1 {
		t.Fatalf("insert after query failed: len=%d, want %d",
			tr.Len(), before+1)
	}
}

// Case 1b: callback deletes from the SAME tree via Scan -> ErrQueryReentry.
func TestQueryReentryDeleteSameTree(t *testing.T) {
	tr := makeProtectedTree()
	before := tr.Len()
	err := tr.ScanE(func(min, max [2]float64, data int) bool {
		tr.Delete(min, max, data)
		return true
	})
	if !errors.Is(err, ErrQueryReentry) {
		t.Fatalf("expected ErrQueryReentry, got %v", err)
	}
	if tr.Len() != before {
		t.Fatalf("reentrant delete was not rejected: len=%d, want %d",
			tr.Len(), before)
	}
	t.Logf("delete-during-scan: err=%v len=%d (unchanged)", err, tr.Len())

	// protection released: delete works after the query
	tr.Delete([2]float64{0, 0}, [2]float64{0, 0}, 0)
	if tr.Len() != before-1 {
		t.Fatalf("delete after query failed: len=%d, want %d",
			tr.Len(), before-1)
	}
}

// Case 1c: callback replaces in the SAME tree via Nearby -> ErrQueryReentry.
func TestQueryReentryNearbySameTree(t *testing.T) {
	tr := makeProtectedTree()
	before := tr.Len()
	err := tr.NearbyE(
		BoxDist[float64, int]([2]float64{0, 0}, [2]float64{0, 0}, nil),
		func(min, max [2]float64, data int, dist float64) bool {
			tr.Replace(min, max, data, min, max, data+1000)
			return true
		})
	if !errors.Is(err, ErrQueryReentry) {
		t.Fatalf("expected ErrQueryReentry, got %v", err)
	}
	if tr.Len() != before {
		t.Fatalf("reentrant replace was not rejected: len=%d, want %d",
			tr.Len(), before)
	}
	t.Logf("replace-during-nearby: err=%v len=%d (unchanged)", err, tr.Len())

	// protection released: replace works after the query
	tr.Replace([2]float64{0, 0}, [2]float64{0, 0}, 0,
		[2]float64{0, 0}, [2]float64{0, 0}, 1000)
	var found bool
	tr.Search([2]float64{0, 0}, [2]float64{0, 0},
		func(min, max [2]float64, data int) bool {
			if data == 1000 {
				found = true
			}
			return true
		})
	if !found {
		t.Fatal("replace after query failed")
	}
}

// Case 2: callback mutates ANOTHER independent tree -> allowed, no error.
func TestQueryReentryOtherTreeAllowed(t *testing.T) {
	tr := makeProtectedTree()
	other := new(RTreeG[int])
	var visited int
	err := tr.SearchE([2]float64{0, 0}, [2]float64{9, 9},
		func(min, max [2]float64, data int) bool {
			visited++
			other.Insert(min, max, data) // different tree: allowed
			return true
		})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if other.Len() != visited || visited == 0 {
		t.Fatalf("other tree len=%d, visited=%d", other.Len(), visited)
	}
	if tr.Len() != 10 {
		t.Fatalf("queried tree changed: len=%d", tr.Len())
	}
	t.Logf("other-tree insert: err=%v visited=%d otherLen=%d",
		err, visited, other.Len())
}

// Case 3: callback stops early -> no error, protection released.
func TestQueryReentryEarlyStop(t *testing.T) {
	tr := makeProtectedTree()
	var visited int
	err := tr.SearchE([2]float64{0, 0}, [2]float64{9, 9},
		func(min, max [2]float64, data int) bool {
			visited++
			return false // stop immediately
		})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if visited != 1 {
		t.Fatalf("expected exactly 1 visit, got %d", visited)
	}
	// protection must be released after early stop
	tr.Insert([2]float64{50, 50}, [2]float64{50, 50}, 50)
	if tr.Len() != 11 {
		t.Fatalf("insert after early stop failed: len=%d", tr.Len())
	}
	t.Logf("early-stop: err=%v visited=%d, post-query insert ok (len=%d)",
		err, visited, tr.Len())
}

// Case 3b: callback panics -> protection must still be released.
func TestQueryReentryCallbackPanic(t *testing.T) {
	tr := makeProtectedTree()
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic from callback")
			}
		}()
		_ = tr.SearchE([2]float64{0, 0}, [2]float64{9, 9},
			func(min, max [2]float64, data int) bool {
				panic("boom")
			})
	}()
	// protection must be released even after a panic
	tr.Insert([2]float64{50, 50}, [2]float64{50, 50}, 50)
	if tr.Len() != 11 {
		t.Fatalf("insert after panic failed: len=%d", tr.Len())
	}
	// and a subsequent protected query works normally
	err := tr.ScanE(func(min, max [2]float64, data int) bool { return true })
	if err != nil {
		t.Fatalf("expected nil error after panic recovery, got %v", err)
	}
	t.Log("panic-in-callback: protection released, tree usable afterwards")
}

// Default mode: protection disabled -> original interface and behavior.
func TestQueryProtectionDisabledByDefault(t *testing.T) {
	tr := new(RTreeG[int])
	if tr.QueryProtection() {
		t.Fatal("protection must be disabled by default")
	}
	for i := 0; i < 5; i++ {
		x := float64(i)
		tr.Insert([2]float64{x, x}, [2]float64{x, x}, i)
	}
	// Mutating the tree from a callback is not reported when the option
	// is off (legacy behavior, no error surfaced).
	err := tr.SearchE([2]float64{0, 0}, [2]float64{4, 4},
		func(min, max [2]float64, data int) bool {
			tr.Insert([2]float64{100, 100}, [2]float64{100, 100}, 100)
			return false // stop so the legacy mutation stays bounded
		})
	if err != nil {
		t.Fatalf("expected nil error with protection disabled, got %v", err)
	}
	// Original void-returning interface still works unchanged.
	var count int
	tr.Search([2]float64{0, 0}, [2]float64{100, 100},
		func(min, max [2]float64, data int) bool {
			count++
			return true
		})
	if count != tr.Len() {
		t.Fatalf("search visited %d, len=%d", count, tr.Len())
	}
	t.Logf("protection-off: err=%v len=%d (legacy behavior preserved)",
		err, tr.Len())
}

// Temp-dir fixture: run the three acceptance scenarios and write a small
// report file demonstrating the actual results.
func TestQueryReentryReport(t *testing.T) {
	dir := t.TempDir()
	report := filepath.Join(dir, "reentry-report.txt")
	var out []string
	logf := func(format string, args ...any) {
		out = append(out, fmt.Sprintf(format, args...))
	}

	// scenario 1: mutate same tree during query
	tr1 := makeProtectedTree()
	err1 := tr1.SearchE([2]float64{0, 0}, [2]float64{9, 9},
		func(min, max [2]float64, data int) bool {
			tr1.Insert([2]float64{9, 9}, [2]float64{9, 9}, 999)
			tr1.Delete(min, max, data)
			return true
		})
	logf("same-tree mutation during query: err=%v len=%d", err1, tr1.Len())
	tr1.Insert([2]float64{9, 9}, [2]float64{9, 9}, 999)
	logf("after query, insert same tree: ok len=%d", tr1.Len())

	// scenario 2: mutate another tree during query
	tr2 := makeProtectedTree()
	other := new(RTreeG[int])
	err2 := tr2.ScanE(func(min, max [2]float64, data int) bool {
		other.Insert(min, max, data)
		return true
	})
	logf("other-tree mutation during query: err=%v otherLen=%d",
		err2, other.Len())

	// scenario 3: early stop
	tr3 := makeProtectedTree()
	var visited int
	err3 := tr3.ScanE(func(min, max [2]float64, data int) bool {
		visited++
		return false
	})
	tr3.Delete([2]float64{0, 0}, [2]float64{0, 0}, 0)
	logf("early stop: err=%v visited=%d, post-query delete ok len=%d",
		err3, visited, tr3.Len())

	if !errors.Is(err1, ErrQueryReentry) || err2 != nil || err3 != nil {
		t.Fatalf("unexpected results: %v %v %v", err1, err2, err3)
	}
	var content string
	for _, ln := range out {
		content += ln + "\n"
	}
	if err := os.WriteFile(report, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(report)
	t.Logf("report fixture %s:\n%s", report, data)
}

// Example demonstrating the optional query reentrancy protection.
func ExampleRTreeG_SetQueryProtection() {
	tr := new(RTreeG[int])
	tr.Insert([2]float64{0, 0}, [2]float64{1, 1}, 1)
	tr.Insert([2]float64{2, 2}, [2]float64{3, 3}, 2)
	tr.SetQueryProtection(true)

	// Mutating the same tree from a query callback is rejected.
	err := tr.ScanE(func(min, max [2]float64, data int) bool {
		tr.Delete(min, max, data)
		return true
	})
	fmt.Println("delete during scan:", err)
	fmt.Println("len after rejected delete:", tr.Len())

	// Protection is released once the query finishes.
	tr.Delete([2]float64{0, 0}, [2]float64{1, 1}, 1)
	fmt.Println("len after delete post-query:", tr.Len())

	// Output:
	// delete during scan: rtree: cannot mutate the tree from within an active query callback
	// len after rejected delete: 2
	// len after delete post-query: 1
}
