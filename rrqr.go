package dimred

import (
	"fmt"
	"log"
	//"math"
	"os"
	//"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/brookluers/dstream/dstream"
)

const (
	defaultRRQRTol float64 = 1e-6
)

// NewRRQR returns an allocated RRQR value for the given dstream.
// All columns not specified in a call to Keep must have float64 type.
func NewRRQR(data dstream.Dstream) *RRQR {

	return &RRQR{
		data: data,
	}
}

// RRQR identifies a maximal set of linearly independent columns in a dstream,
// using a rank-revealing Cholesky decomposition.
type RRQR struct {

	// The Gram matrix of the columns being assessed for linear dependence
	cpr []float64

	// The input dstream
	data dstream.Dstream

	// The output dstream
	rdata dstream.Dstream

	// The tolerance parameter for dropping terms.
	tol float64

	// Variables that we will keep regardless (ignore these in the check for linear dependence).
	keep    []string
	keeppos []int

	// A logger, not used if nil
	log *log.Logger

	// Positions to consider in the check for linear dependence (everything not in keeppos).
	checkpos []int
}

func (fr *RRQR) init() {

	vpos := make(map[string]int)
	for k, v := range fr.data.Names() {
		vpos[v] = k
	}

	km := make(map[string]bool)
	for _, s := range fr.keep {
		ii, ok := vpos[s]
		if !ok {
			msg := fmt.Sprintf("Variable '%s' specified in RRQR.Keep is not in the dstream.", s)
			panic(msg)
		}
		fr.keeppos = append(fr.keeppos, ii)
		km[s] = true
	}

	for k, n := range fr.data.Names() {
		if !km[n] {
			fr.checkpos = append(fr.checkpos, k)
		}
	}
}

// getcpr calculates the cross product matrix of the variables being checked.
func (fr *RRQR) getcpr() {

	p := len(fr.checkpos)
	cpr := make([]float64, p*p)

	fr.data.Reset()
	for fr.data.Next() {
		vars := make([][]float64, p)

		for j, k := range fr.checkpos {
			vars[j] = fr.data.GetPos(k).([]float64)
		}

		for j1 := 0; j1 < p; j1++ {
			for j2 := 0; j2 <= j1; j2++ {
				u := floats.Dot(vars[j1], vars[j2])
				cpr[j1*p+j2] += u
				if j1 != j2 {
					cpr[j2*p+j1] += u
				}
			}
		}
	}

	fr.cpr = cpr
}

func (fr *RRQR) DimCheck() int {
	return len(fr.checkpos)
}

func (fr *RRQR) CPR() []float64 {
	return fr.cpr
}

// Tol sets the tolerance parameter for dropping columns that are nearly
// linearly dependent with the other columns.
func (fr *RRQR) Tol(tol float64) *RRQR {
	fr.tol = tol
	return fr
}

// LogFile specifies the name of a file used for logging.
func (fr *RRQR) LogFile(filename string) *RRQR {
	f, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	fr.log = log.New(f, "", log.Lshortfile)

	return fr
}

// Done completes configuration.
func (fr *RRQR) Done() *RRQR {

	if fr.tol == 0 {
		fr.tol = defaultRRQRTol
	}

	fr.init()
	fr.getcpr()
	if fr.log != nil {
		fr.log.Printf(fmt.Sprintf("cross product matrix: \n%v\n\n", fr.cpr))
	}

	p := len(fr.checkpos)
	rank, jpr := pivotqr(fr.cpr, p, p, fr.tol, fr.log)
	fmt.Printf("RRQR: numerical rank = %v\n", rank)
	fmt.Printf("RRQR: final permutation = %v\n", jpr)

	if fr.log != nil {
		msg := "a log message from RRQR"
		fr.log.Printf(msg)
	}

	if fr.log != nil {
		msg := fmt.Sprintf("Retained %d out of %d variables", rank, p)
		fr.log.Printf(msg)
	}

	rpos := make(map[int]bool)
	for _, k := range jpr {
		rpos[fr.checkpos[k]] = true
	}

	for _, k := range fr.keeppos {
		rpos[k] = true
	}

	var drop []string
	names := fr.data.Names()
	for k := range names {
		if !rpos[k] {
			drop = append(drop, names[k])
		}
	}
	fmt.Printf("RRQR: dropping columns %v\n", drop)
	fr.rdata = dstream.DropCols(fr.data, drop...)

	return fr
}

// Data returns the dstream constructed by RRQR.
func (fr *RRQR) Data() dstream.Dstream {
	return fr.rdata
}

// Keep specifies variables that are not considered in the linear
// independence assessment.
func (fr *RRQR) Keep(vars ...string) *RRQR {

	fr.keep = vars
	return fr
}

func pivotqr(a []float64, m int, n int, tol float64, frlog *log.Logger) (int, []int) {

	// Initial permutation (identity)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}

	// Identity matrix
	idn := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		idn.Set(i, i, 1)
	}

	k := 0
	n_k := n - k

	// Initialize R = input matrix
	R := mat.NewDense(m, n, a) 
	var qr mat.QR
	qr.Factorize(R)
	Ck := qr.RTo(nil) // Initially, k=0 and Ck is the entire qr.R(R)
	
	// Column norms of Ck
	CkColNorms := make([]float64, n_k)
	CkNormSorted := make([]float64, n_k)
	var ccol mat.Vector
	for i := 0; i < n_k; i++ {
		ccol = Ck.ColView(i)
		CkColNorms[i] = mat.Norm(ccol, 2)
		CkNormSorted[i] = CkColNorms[i]
	}
	if frlog != nil {
		msg := fmt.Sprintf("CkColNorms = %v\n", CkColNorms)
		frlog.Printf(msg)
	}
	CkNormOrder := make([]int, n_k)
	floats.Argsort(CkNormSorted, CkNormOrder)
	if frlog != nil {
		frlog.Printf(fmt.Sprintf("Sorted Norms: %v\nSorted norm indices: %v\n", CkNormSorted, CkNormOrder))
		Ckfmt := mat.Formatted(Ck, mat.Prefix("    "), mat.Squeeze())
		frlog.Printf(fmt.Sprintf("Ck = \n%v\n\n", Ckfmt))
	}

	// Permutation matrix
	pimat := mat.NewDense(n, n, nil)
	pimat.Clone(idn)
	var jmax int // index of largest column norm of Ck
	if frlog != nil {
		frlog.Printf(fmt.Sprintf("\n---------Starting pivoted QR iterations---------\n\n"))
	}
	// Largest column norm of Ck
	maxNorm := CkColNorms[CkNormOrder[len(CkNormOrder)-1]]
	for maxNorm > tol {
		jmax = CkNormOrder[len(CkNormOrder)-1]
		k++
		n_k = n - k

		swaps := make([]int, n)
		prevperm := make([]int, n)
		copy(prevperm, perm)
		for i := 0; i < n; i++ {
			swaps[i] = i
			if i == k-1 {
				perm[i] = prevperm[k+jmax-1]
				swaps[i] = k + jmax - 1
			} else if i == k+jmax-1 {
				swaps[i] = k - 1
				perm[i] = prevperm[k-1]
			}
		}

		switchcols := mat.NewDense(n, n, nil)
		switchcols.Permutation(n, swaps)

		toFactorize := mat.NewDense(m, n, nil)
		toFactorize.Mul(R, switchcols)
		qr.Factorize(toFactorize)
		R.Clone(qr.RTo(nil)) //  R = R_k ( R * switchcols)

		pimat.Mul(pimat, switchcols)
		qr.Factorize(R)

		if frlog != nil {
			frlog.Printf(fmt.Sprintf("jmax = %v\n", jmax))
			frlog.Printf(fmt.Sprintf("swaps = %v\n", swaps))
			swfmt := mat.Formatted(switchcols, mat.Prefix(""), mat.Squeeze())
			frlog.Printf(fmt.Sprintf("switchcols = \n%v\n\n", swfmt))
			Rfmt := mat.Formatted(R, mat.Prefix(" "), mat.Squeeze())
			frlog.Printf(fmt.Sprintf("R = \n%v\n\n", Rfmt))
			frlog.Printf(fmt.Sprintf("Computing Cknext, m=%v, k=%v, n=%v\n", m, k, n))
		}

		Cknext := mat.DenseCopyOf(qr.RTo(nil).Slice(k, m, k, n))
		Ckfmt := mat.Formatted(Cknext, mat.Prefix("   "), mat.Squeeze())

		CkColNorms = CkColNorms[0:n_k]
		CkNormSorted = CkNormSorted[0:n_k]
		for i := 0; i < n_k; i++ {
			ccol = Cknext.ColView(i)
			CkColNorms[i] = mat.Norm(ccol, 2)
			CkNormSorted[i] = CkColNorms[i]
		}
		CkNormOrder = CkNormOrder[0:n_k]
		floats.Argsort(CkNormSorted, CkNormOrder)

		maxNorm = CkColNorms[CkNormOrder[len(CkNormOrder)-1]]
		if frlog != nil {
			frlog.Printf(fmt.Sprintf("Unsorted Norms: %v\nSorted Norms: %v\nSorted norm indices: %v\n", CkColNorms, CkNormSorted, CkNormOrder))
			frlog.Printf(fmt.Sprintf("maxNorm = %v", maxNorm))
			frlog.Printf(fmt.Sprintf("Cknext = \n%v\n\n", Ckfmt))
		}

	}

	if frlog != nil {
		frlog.Printf(fmt.Sprintf("\n--------Finished pivoted QR iterations------\n\n"))
		pimatfmt := mat.Formatted(pimat, mat.Prefix("   "), mat.Squeeze())
		frlog.Printf(fmt.Sprintf("pimat = \n%v\n\n", pimatfmt))
	}

	return k, perm[0:k]
}
