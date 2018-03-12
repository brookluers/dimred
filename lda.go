package dimred

import (
	"fmt"
	"log"
	"math"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/brookluers/dstream/dstream"
)

type LDA struct {
	*MomentStream

	// The min(k-1, p) discriminant directions
	ldirs [][]float64
	
	// The within-class covariance estimate
	wmat []float64

	// The between-class covariance estimate (covariance of the means)
	bmat []float64

	// The proportion of samples in each class
	pihat []float64
	
	// The number of classes
	k int

	// The number of LDA directions
	ndir int

}

// LDir returns an estimated discriminant direction
// for use in LDA classification or dimension reduction for optimal class separation
func (lda *LDA) LDir(j int) []float64 {
	return lda.ldirs[j]
}

// NewLDA returns a new LDA value for the given data stream.
func NewLDA(data dstream.Dstream, response string, nclass int) *LDA {

	d := &LDA {
		MomentStream: NewMomentStream(data, response),
		k: nclass,
	}
	return d
}

func (lda *LDA) Done() *LDA {
	lda.MomentStream.Done()
	lda.calcMargMean()
	lda.calcMargCov()    
	lda.calcPi()
	lda.calcWithinCov()
	lda.calcBetweenCov()
	if lda.log != nil {
	   lda.log.Printf("proportion in each class: %v\n", lda.GetPi())
	   lda.log.Printf("Nobs = %v\n", lda.Data.NumObs())
	}
	return lda
}

func (lda *LDA) calcPi() *LDA {
     pi := make([]float64, lda.k)
     for j := 0; j < lda.k; j++{
     	 pi[j] = float64(lda.GetN(j)) / float64(lda.Data.NumObs())
     }
     lda.pihat = pi

     return lda
}

// GetPi returns the sample proportions for each class
func (lda *LDA) GetPi() []float64 {
     return lda.pihat
}

// calcWithinCov computes the within-class covariance matrix
func (lda *LDA) calcWithinCov() *LDA {
     p := lda.Dim()
     pp := p * p
     N := lda.Data.NumObs()
     w := make([]float64, pp)

     for j := 0; j < lda.k; j++{
	 floats.AddScaled(w, float64(lda.GetN(j)), lda.GetCov(j))
     }
     floats.Scale(1/float64(N - lda.k), w)

     lda.wmat = w

     return lda
}

func (lda *LDA) GetNDir() int {
     return lda.ndir
}

func (lda *LDA) GetWCov() []float64 {
     return lda.wmat
}

func (lda *LDA) GetBCov() []float64 {
     return lda.bmat
}

func (lda *LDA) calcBetweenCov() *LDA {

     // the between-class covariance matrix
     p := lda.Dim()
     pp := p * p
     N := lda.Data.NumObs()
     bb :=  make([]float64, pp)

     for j := 0; j < lda.k; j++{
     	 classmean := lda.GetMean(j)
	 Nk := lda.GetN(j)
     	 for k1 := 0; k1 < p; k1++ {
	     u1 := classmean[k1] - lda.GetMargMean()[k1]
	     for k2 := 0; k2 <= k1; k2++ {
	     	 u2 := classmean[k2] - lda.GetMargMean()[k2]
		 bb[k1*p+k2] += float64(Nk) * u1 * u2
	     }
	 }
     }
     reflect(bb, p)
     floats.Scale(1/float64(N), bb)
     lda.bmat = bb

     return lda
}

func flipMat(p int) *mat.Dense {
     m := mat.NewDense(p, p, nil)
     colpos := make([]int, p)
     for i := 0; i < p; i++{
     	 colpos[i] = p - i - 1
     	 for j := 0; j < p; j++{
	     if j==colpos[i] {
	     	m.Set(i, j, 1)
	     } else {
	       m.Set(i, j, 0)
	     }
	 }
     }
     return m
}

func min(a, b int) int {
     if a < b {
     	return a
     }
     return b
}

func diagMat(v []float64) *mat.Dense {
	n := len(v)
	m := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		m.Set(i, i, v[i])
	}
	return m
}

func (lda *LDA) Fit() {

	p := lda.Dim()
	lda.ndir = min(lda.k - 1, p)
	if lda.log != nil {
	   lda.log.Printf("ndir = %v\n", lda.ndir)
	   lda.log.Printf("p = %v\n", p)
	}

	weig := new(mat.EigenSym)
	ww := mat.NewSymDense(p, lda.GetWCov())
	bb := mat.NewSymDense(p, lda.GetBCov())
	
	ok := weig.Factorize(ww, true)
	if !ok {
	   panic("can't factorize within-class covariance")
	}
	wevals := weig.Values(nil)
	wevec := mat.NewDense(p, p, nil)
	wevec.EigenvectorsSym(weig)
	
	wU := mat.NewDense(p, p, nil)
	pFlip := flipMat(p)
	wU.Mul(wevec, pFlip)
	// wU_fmt := mat.Formatted(wU, mat.Prefix("    "), mat.Squeeze())
	// fmt.Printf("wU = %v\n", wU_fmt)
	
	wev_inv2 := make([]float64, p)
	for j := 0; j < p; j++{
	    wev_inv2[p - j - 1] = math.Pow(wevals[j], -0.5)
	}
	// fmt.Printf("inverse square root of eigenvalues: %v\n\n", wev_inv2)
	di2 := diagMat(wev_inv2)
	// di2_fmt := mat.Formatted(di2, mat.Prefix("    "), mat.Squeeze())
	// fmt.Printf("di2 = %v\n\n", di2_fmt)
	wi2 := mat.NewDense(p, p, nil)
	wi2.Mul(di2, wU.T())
	// wi2_fmt := mat.Formatted(wi2, mat.Prefix("    "), mat.Squeeze())
	// fmt.Printf("wi2 = %v\n\n", wi2_fmt)
	t1 := mat.NewDense(p, p, nil)
	t1.Mul(wi2, bb)
	bstar := mat.NewDense(p, p, nil)
	bstar.Mul(t1, wi2.T())
	// bstar_fmt := mat.Formatted(bstar, mat.Prefix("     "), mat.Squeeze())
	// fmt.Printf("Bstar = %v\n\n", bstar_fmt)
	t2 := make([]float64, p*p)
	for i := 0; i < p; i++{
	    for j := 0; j < p; j++{
	    	t2[i * p + j] = bstar.At(i, j)
	    }
	}
	bstarsym := mat.NewSymDense(p, t2)
	bstar_eig := new(mat.EigenSym)
	ok = bstar_eig.Factorize(bstarsym, true)
	if !ok {
	   panic("can't factorize sphered between-class covariance")
	}
	bstar_evec := mat.NewDense(p, p, nil)
	bstar_evec.EigenvectorsSym(bstar_eig)
	bstar_evec.Mul(bstar_evec, pFlip)

	bstar_sliced_T := bstar_evec.Slice(0, p, 0, lda.ndir).T()
	bstar_evec_fmt := mat.Formatted(bstar_sliced_T, 
		       mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("bstar_evec.slice...T() = %v\n\n", bstar_evec_fmt)
	r, c := bstar_sliced_T.Dims()
	fmt.Printf("...has dimensions %v by %v \n", r, c)
	wi2_fmt := mat.Formatted(wi2.T(), mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("wi2.T() = %v\n", wi2_fmt)
	r, c = wi2.T().Dims()
	fmt.Printf("...has dimensions %v by %v\n", r, c)

	
	coord := mat.NewDense(lda.ndir, p, nil)
	r, c = coord.Dims()
	fmt.Printf("coord has dimensions %v by %v\n", r, c)

	coord.Mul(bstar_evec.Slice(0, p, 0, lda.ndir).T(), wi2.T())
	
	lda.ldirs = make([][]float64, lda.ndir)
	for j := 0; j < lda.ndir; j++{
	    lda.ldirs[j] = make([]float64, p)
	    lda.ldirs[j] = coord.RawRowView(j)
	}

	if lda.log != nil {
	   coord_fmt := mat.Formatted(coord, mat.Prefix("    "), mat.Squeeze())
	   wevec_fmt := mat.Formatted(wevec, mat.Prefix("    "), mat.Squeeze())
	   bb_fmt := mat.Formatted(bb, mat.Prefix("     "), mat.Squeeze())
	   lda.log.Printf("B = %v\n\n", bb_fmt)
	   ww_fmt := mat.Formatted(ww, mat.Prefix("     "), mat.Squeeze())
	   lda.log.Printf("W = %v\n\n", ww_fmt)
	   lda.log.Printf("Eigenvectors of W = %v\n\n", wevec_fmt)
	   lda.log.Printf("W eigenvalues: %v\n\n", wevals)
	   lda.log.Printf("coord = %v\n\n", coord_fmt)
	   lda.log.Printf("ndir = %v\n", lda.ndir)
	   lda.log.Printf("lda.ldirs = %v\n", lda.ldirs)	
	}
}

func (lda *LDA) SetLogFile(filename string) *LDA {
	fid, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	lda.log = log.New(fid, "", log.Lshortfile)

	return lda
}
