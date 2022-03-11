use numpy::{PyArray3, PyReadonlyArray3};
//use ndarray::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyFunction;
use voxcov as vc;
use voxcov::blur::AddGaussianToMap;
use voxcov::coverage::{AddCoverage, DelCoverage, MapLike};

#[pyclass]
struct SMOC {
    inner: vc::score::SMOC<vc::coverage::DiffCoverageMap>,
    exp: vc::blur::RawMap,
    sim: vc::blur::RawMap,
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
struct CoverageMap {
    inner: vc::coverage::CoverageMap,
}

#[pyclass]
struct BlurMap {
    map: vc::blur::Map,
    blurrer: vc::blur::GaussianBlur,
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
            &mut |x: [usize; 3]| {
                ()
            },
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

    fn add_sphere(&mut self, point: [f64; 3], radius: f64, f: &PyFunction) -> usize {
        self.inner.add_cov(
            &vc::geom::Sphere::new(point, radius),
            &mut |x: [usize; 3]| {
                f.call1((x,)).expect("Failed to call python function");
            },
        )
    }

    fn del_sphere(&mut self, point: [f64; 3], radius: f64, f: &PyFunction) -> usize {
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

    fn add_guassian(&mut self, point: [f64; 3], height: f64, width: f64) {
        self.blurrer.add_gaussian(&mut self.map, &point, height, width)
    }
}

#[pymodule]
fn voxcov(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SMOC>()?;
    m.add_class::<SMOCX>()?;
    m.add_class::<SMOCN>()?;
    m.add_class::<DiffCoverageMap>()?;
    m.add_class::<CoverageMap>()?;
    m.add_class::<BlurMap>()?;
    Ok(())
}
