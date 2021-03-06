package dimred

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/brookluers/dstream/dstream"
)

// arspec specifies a test population for simulation.  The
// observations alternate between the two groups.
type arspec struct {

	// Sample size
	n int

	// Number of variables
	p int

	// Autocorrelation parameter
	r float64

	// Transformation function that can be used to introduce a
	// mean. The arguments are i (subject), j (variable) and e,
	// where e is the AR-1 error term.
	xform func(int, int, float64) float64
}

func docdat1(chunksize int, ar arspec) dstream.Dstream {

	rand.Seed(384293)

	n := ar.n
	p := ar.p
	r := ar.r
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1) // first element is group label 0/1

	// First variable is all white
	for i := 0; i < n; i++ {
		da[0] = append(da[0], rand.NormFloat64())
	}

	// Autocorrelated noise
	for j := 1; j < p+1; j++ {
		for i := 0; i < n; i++ {
			da[j] = append(da[j], r*da[j-1][i]+rc*rand.NormFloat64())
		}
	}

	// Mean structure
	if ar.xform != nil {
		for i := 0; i < n; i++ {
			for j := 0; j < p+1; j++ {
				da[j][i] = ar.xform(i, j, da[j][i])
			}
		}
	}

	// Create the group labels
	for i := 0; i < n; i++ {
		da[0][i] = float64(i % 2)
	}

	// Create a dstream
	var ida [][]interface{}
	for _, x := range da {
		ida = append(ida, []interface{}{x})
	}
	na := []string{"y"}
	for j := 0; j < p; j++ {
		na = append(na, fmt.Sprintf("x%d", j+1))
	}
	ds := dstream.NewFromArrays(ida, na)
	ds = dstream.MaxChunkSize(ds, chunksize)

	return ds
}

func TestDOC1(t *testing.T) {

	xform := func(i int, j int, x float64) float64 {
		if j == 0 {
			return float64(i % 2) // group label
		}
		// Mean and variance effect.  Group 2 variance is
		// four times the group 1 variance.
		z := float64(i%2)*float64(j) + float64(i%2+1)*x
		return z
	}

	ar := arspec{n: 10000, p: 5, r: 0.6, xform: xform}

	chunksize := 1000
	da := docdat1(chunksize, ar)

	doc := NewDOC(da, "y").SetLogFile("ss").Done()
	doc.Fit(4)

	// Check the means
	m0 := doc.GetMean(0)
	mx := make([]float64, len(m0))
	if !floats.EqualApprox(m0, mx, 0.02) {
		t.Fail()
	}
	for k := range mx {
		mx[k] = float64(k + 1)
	}
	m1 := doc.GetMean(1)
	if !floats.EqualApprox(m1, mx, 0.05) {
		t.Fail()
	}

	// Check the covariances
	c0 := doc.GetCov(0)
	c1 := doc.GetCov(1)
	rv := make([]float64, len(c0))
	floats.DivTo(rv, c1, c0)
	r := floats.Sum(rv) / float64(len(rv))
	if math.Abs(r-4) > 0.1 {
		t.Fail()
	}

	// Check the eigenvalues
	for _, e := range doc.Eig() {
		if math.Abs(1.2-math.Abs(e)) > 0.05 {
			t.Fail()
		}
	}
}

func TestDOC2(t *testing.T) {

	ar := arspec{n: 10000, p: 5, r: 0.6}
	da := docdat1(1000, ar)

	doc := NewDOC(da, "y").SetLogFile("s2").Done()
	doc.Fit(4)

	_ = doc // Just a smoke test
}

// Generate data from a forward regression model.  X marginally is AR,
// Y|X depends only on one linear function of X.
func docdat3(chunksize int) dstream.Dstream {

	n := 10000
	p := 5
	r := 0.6
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1)

	for i := 0; i < n; i++ {
		da[0] = append(da[0], rand.NormFloat64())
	}
	for j := 1; j < p+1; j++ {
		for i := 0; i < n; i++ {
			da[j] = append(da[j], r*da[j-1][i]+rc*rand.NormFloat64())
		}
	}

	for i := 0; i < n; i++ {
		u := da[1][i] - 2*da[2][i] + 3*da[4][i]
		if u > 0 {
			da[0][i] = 1
		} else {
			da[0][i] = 0
		}
	}

	na := []string{"y"}
	for j := 0; j < p; j++ {
		na = append(na, fmt.Sprintf("x%d", j+1))
	}

	var ida [][]interface{}
	for _, x := range da {
		ida = append(ida, []interface{}{x})
	}

	df := dstream.NewFromArrays(ida, na)
	df = dstream.MaxChunkSize(df, chunksize)

	return df
}

func TestDOC3(t *testing.T) {

	df := docdat3(1000)

	doc := NewDOC(df, "y").SetLogFile("s3").Done()
	doc.Fit(2)

	_ = doc // Just a smoke test
}

func TestProj(t *testing.T) {

	xform := func(i, j int, x float64) float64 {
		if j <= 2 {
			return 2 * x
		}
		return x
	}

	ar := arspec{n: 10000, p: 5, r: 0.6, xform: xform}
	df := docdat1(1000, ar)

	doc := NewDOC(df, "y").SetLogFile("sp").SetProjection(2).Done()
	doc.Fit(2)
}
