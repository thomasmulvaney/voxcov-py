use numpy::{PyArray, PyArray3, PyReadonlyArray3};
//use ndarray::prelude::*;
use ::voxcov as vc;
use ::voxcov::bfactor::Mixture;
use ::voxcov::blur::AddGaussianToMap;
use ::voxcov::coverage::{AddCoverage, DelCoverage, MapLike};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyFunction;

#[pyclass]
struct SMOC {
    inner: vc::score::SMOC<vc::coverage::DiffCoverageMap>,
    exp: vc::blur::RawMap,
    sim: vc::blur::RawMap,
}

#[pyclass]
struct MixtureModel {
    inner: vc::bfactor::Mixture,
}

#[pyclass]
struct SMI {
    inner: vc::score::SMI<vc::coverage::DiffCoverageMap>,
    exp: vc::blur::RawMap,
    sim: vc::blur::RawMap,
}
#[pyclass]
struct SMIX {
    inner: vc::score::SMI<vc::coverage::DiffCoverageMap>,
    exp: vc::blur::RawMap,
}

#[pyclass]
struct SMOCX {
    inner: vc::score::SMOC<vc::coverage::DiffCoverageMap>,
    exp: vc::blur::RawMap,
}

#[pyclass]
struct SMOCN {
    inner: vc::score::SMOC<vc::coverage::NaiveDiffCoverageMap>,
    exp: vc::blur::RawMap,
}

#[pyclass]
struct DiffCoverageMap {
    inner: vc::coverage::DiffCoverageMap,
}

#[pyclass]
struct NaiveDiffCoverageMap {
    inner: vc::coverage::NaiveDiffCoverageMap,
}

#[pyclass]
struct CoverageMap {
    inner: vc::coverage::CoverageMap,
}

#[pyclass]
struct BlurMap {
    map: vc::blur::Map,
    blurrer: vc::blur::GaussianBlur,
}

#[pymethods]
impl MixtureModel {
    #[new]
    fn init(
        apix: [f64; 3],
        origin: [f64; 3],
        size: [usize; 3],
        exp: PyReadonlyArray3<f64>,
    ) -> Self {
        let map = vc::blur::Map::new(origin, apix, size);
        let rmap = vc::blur::RawMap::new(origin, apix, size, exp.to_vec().unwrap());
        Self {
            inner: Mixture::new(map, rmap),
        }
    }

    fn add_gaussian(&mut self, mean: [f64; 3], sigma: f64, weight: f64) {
        self.inner.add_gaussian(mean, sigma, weight);
    }

    fn step(&mut self) {
        self.inner.step()
    }

    fn sigma(&self, idx: usize) -> f64 {
        self.inner.sigma(idx)
    }

    fn set_sigma(&mut self, idx: usize, sigma: f64) {
        self.inner.set_sigma(idx, sigma);
    }

    fn set_mean(&mut self, idx: usize, mean: [f64; 3]) {
        self.inner.set_mean(idx, mean);
    }
    fn mean(&self, idx: usize) -> [f64; 3] {
        self.inner.mean(idx)
    }

    fn logprob(&self) -> f64 {
        0f64
    }

    fn logprob_idx(&self, idx: usize) -> f64 {
        0f64
    }

    fn noise(&self) -> f64 {
        self.inner.noise()
    }

    fn moc(&self) -> f64 {
        self.inner.moc()
    }

    fn set_noise(&mut self, noise: f64) {
        self.inner.set_noise(noise)
    }

    fn to_numpy(self_: PyRefMut<Self>) -> Bound<'_, PyArray3<f64>> {
        let py = self_.py();
        let mut n = 0;
        unsafe {
            let mut res = PyArray::new(
                py,
                (
                    self_.inner.data.densities.size[2],
                    self_.inner.data.densities.size[1],
                    self_.inner.data.densities.size[0],
                ),
                false,
            );
            for i in 0..self_.inner.data.densities.size[0] {
                for j in 0..self_.inner.data.densities.size[1] {
                    for k in 0..self_.inner.data.densities.size[2] {
                        let v = self_.inner.data.densities.grid[n];
                        *res.uget_mut((k, j, i)) = v;
                        n += 1;
                    }
                }
            }
            res.to_owned()
        }
    }
}

#[pymethods]
impl SMI {
    #[new]
    fn init(
        apix: [f64; 3],
        origin: [f64; 3],
        width: usize,
        size: [usize; 3],
        exp: PyReadonlyArray3<f64>,
        sim: PyReadonlyArray3<f64>,
        x_low: f64,
        x_high: f64,
        y_low: f64,
        y_high: f64,
        x_bins: usize,
        y_bins: usize,
    ) -> Self {
        let exp = exp.to_vec().unwrap();
        let sim = sim.to_vec().unwrap();
        Self {
            inner: vc::score::SMI::<vc::coverage::DiffCoverageMap>::new(
                apix, origin, width, x_low, x_high, y_low, y_high, x_bins, y_bins,
            ),
            exp: vc::blur::RawMap::new(apix, origin, size, exp),
            sim: vc::blur::RawMap::new(apix, origin, size, sim),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64) {
        self.inner
            .add_cov(&vc::geom::Sphere::new(point, radius), &self.sim, &self.exp);
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64) {
        self.inner
            .del_cov(&vc::geom::Sphere::new(point, radius), &self.sim, &self.exp);
    }

    fn get_val(&self) -> f64 {
        self.inner.get_val()
    }

    fn zero(&mut self) {
        self.inner.zero()
    }
}

#[pymethods]
impl SMOC {
    #[new]
    fn init(
        apix: [f64; 3],
        origin: [f64; 3],
        width: usize,
        size: [usize; 3],
        exp: PyReadonlyArray3<f64>,
        sim: PyReadonlyArray3<f64>,
    ) -> Self {
        let exp = exp.to_vec().unwrap();
        let sim = sim.to_vec().unwrap();
        Self {
            inner: vc::score::SMOC::<vc::coverage::DiffCoverageMap>::new(apix, origin, width),
            exp: vc::blur::RawMap::new(apix, origin, size, exp),
            sim: vc::blur::RawMap::new(apix, origin, size, sim),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64) {
        self.inner
            .add_cov(&vc::geom::Sphere::new(point, radius), &self.sim, &self.exp);
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64) {
        self.inner
            .del_cov(&vc::geom::Sphere::new(point, radius), &self.sim, &self.exp);
    }

    fn get_val(&self) -> f64 {
        self.inner.get_val()
    }

    fn zero(&mut self) {
        self.inner.zero()
    }
}

#[pymethods]
impl SMIX {
    #[new]
    fn init(
        apix: [f64; 3],
        origin: [f64; 3],
        width: usize,
        size: [usize; 3],
        exp: PyReadonlyArray3<f64>,
        x_low: f64,
        x_high: f64,
        y_low: f64,
        y_high: f64,
        x_bins: usize,
        y_bins: usize,
    ) -> Self {
        let exp = exp.to_vec().unwrap();
        Self {
            inner: vc::score::SMI::<vc::coverage::DiffCoverageMap>::new(
                apix, origin, width, x_low, x_high, y_low, y_high, x_bins, y_bins,
            ),
            exp: vc::blur::RawMap::new(apix, origin, size, exp),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64, sim: &BlurMap) {
        self.inner
            .add_cov(&vc::geom::Sphere::new(point, radius), &sim.map, &self.exp);
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64, sim: &BlurMap) {
        self.inner
            .del_cov(&vc::geom::Sphere::new(point, radius), &sim.map, &self.exp);
    }

    fn get_val(&self) -> f64 {
        self.inner.get_val()
    }

    fn zero(&mut self) {
        self.inner.zero()
    }
}

#[pymethods]
impl SMOCX {
    #[new]
    fn init(
        apix: [f64; 3],
        origin: [f64; 3],
        width: usize,
        size: [usize; 3],
        exp: PyReadonlyArray3<f64>,
    ) -> Self {
        let exp = exp.to_vec().unwrap();
        Self {
            inner: vc::score::SMOC::<vc::coverage::DiffCoverageMap>::new(apix, origin, width),
            exp: vc::blur::RawMap::new(apix, origin, size, exp),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64, sim: &BlurMap) {
        self.inner
            .add_cov(&vc::geom::Sphere::new(point, radius), &sim.map, &self.exp);
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64, sim: &BlurMap) {
        self.inner
            .del_cov(&vc::geom::Sphere::new(point, radius), &sim.map, &self.exp);
    }

    fn get_val(&self) -> f64 {
        self.inner.get_val()
    }

    fn zero(&mut self) {
        self.inner.zero()
    }
}

#[pymethods]
impl SMOCN {
    #[new]
    fn init(
        apix: [f64; 3],
        origin: [f64; 3],
        width: usize,
        size: [usize; 3],
        exp: PyReadonlyArray3<f64>,
    ) -> Self {
        let exp = exp.to_vec().unwrap();
        Self {
            inner: vc::score::SMOC::<vc::coverage::NaiveDiffCoverageMap>::new(apix, origin, width),
            exp: vc::blur::RawMap::new(apix, origin, size, exp),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64, sim: &BlurMap) {
        self.inner
            .add_cov(&vc::geom::Sphere::new(point, radius), &sim.map, &self.exp);
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64, sim: &BlurMap) {
        self.inner
            .del_cov(&vc::geom::Sphere::new(point, radius), &sim.map, &self.exp);
    }

    fn get_val(&self) -> f64 {
        self.inner.get_val()
    }

    fn zero(&mut self) {
        self.inner.zero()
    }
}

#[pymethods]
impl CoverageMap {
    #[new]
    fn init(apix: [f64; 3], origin: [f64; 3], width: usize) -> Self {
        Self {
            inner: vc::coverage::CoverageMap::new(apix, origin, width),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64) -> usize {
        self.inner.add_cov(
            &vc::geom::Sphere::new(point, radius),
            &mut |x: [usize; 3]| (),
        )
    }

    fn zero(&mut self) {
        self.inner.zero();
    }
}

#[pymethods]
impl DiffCoverageMap {
    #[new]
    fn init(apix: [f64; 3], origin: [f64; 3], width: usize) -> Self {
        Self {
            inner: vc::coverage::DiffCoverageMap::new(apix, origin, width),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64, f: &Bound<'_, PyAny>) -> usize {
        self.inner.add_cov(
            &vc::geom::Sphere::new(point, radius),
            &mut |x: [usize; 3]| {
                f.call1((x,)).expect("Failed to call python function");
            },
        )
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64, f: &Bound<'_, PyAny>) -> usize {
        self.inner.del_cov(
            &vc::geom::Sphere::new(point, radius),
            &mut |x: [usize; 3]| {
                f.call1((x,)).expect("Failed to call python function");
            },
        )
    }
}
#[pymethods]
impl NaiveDiffCoverageMap {
    #[new]
    fn init(apix: [f64; 3], origin: [f64; 3], width: usize) -> Self {
        Self {
            inner: vc::coverage::NaiveDiffCoverageMap::new(apix, origin, width),
        }
    }

    fn add_sphere(&mut self, point: [f64; 3], radius: f64, f: &Bound<'_, PyAny>) -> usize {
        self.inner.add_cov(
            &vc::geom::Sphere::new(point, radius),
            &mut |x: [usize; 3]| {
                f.call1((x,)).expect("Failed to call python function");
            },
        )
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64, f: &Bound<'_, PyAny>) -> usize {
        self.inner.del_cov(
            &vc::geom::Sphere::new(point, radius),
            &mut |x: [usize; 3]| {
                f.call1((x,)).expect("Failed to call python function");
            },
        )
    }
}

#[pymethods]
impl BlurMap {
    #[new]
    fn init(apix: [f64; 3], origin: [f64; 3], size: [usize; 3], cutoff: f64) -> Self {
        let res = Self {
            map: vc::blur::Map::new(origin, apix, size),
            blurrer: vc::blur::GaussianBlur::new(cutoff),
        };
        res
    }

    fn zero(&mut self) {
        self.map.zero();
    }

    fn normalize(&mut self) {
        self.map.densities.normalize();
    }

    fn add_gaussian(&mut self, point: [f64; 3], height: f64, width: f64) {
        self.blurrer
            .add_gaussian(&mut self.map, &point, height, width)
    }

    fn to_numpy(self_: PyRefMut<'_, Self>) -> Bound<'_, PyArray3<f64>> {
        let py = self_.py();
        let mut n = 0;
        unsafe {
            let mut res = PyArray::new(
                py,
                (
                    self_.map.densities.size[2],
                    self_.map.densities.size[1],
                    self_.map.densities.size[0],
                ),
                false,
            );
            for i in 0..self_.map.densities.size[0] {
                for j in 0..self_.map.densities.size[1] {
                    for k in 0..self_.map.densities.size[2] {
                        unsafe {
                            let v = self_.map.densities.grid[n];
                            *res.uget_mut((k, j, i)) = v;
                        }
                        n += 1;
                    }
                }
            }
            res.to_owned()
        }
    }
}

#[pymodule]
fn voxcov(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SMI>()?;
    m.add_class::<SMOC>()?;
    m.add_class::<SMOCX>()?;
    m.add_class::<SMOCN>()?;
    m.add_class::<DiffCoverageMap>()?;
    m.add_class::<NaiveDiffCoverageMap>()?;
    m.add_class::<CoverageMap>()?;
    m.add_class::<BlurMap>()?;
    m.add_class::<MixtureModel>()?;
    Ok(())
}
