use daft_logical_plan::LogicalPlanBuilder;
use eyre::{ensure, Context};
use spark_connect::Range;

use super::SparkAnalyzer;

impl SparkAnalyzer<'_> {
    pub fn range(&self, range: Range) -> eyre::Result<LogicalPlanBuilder> {
        use daft_scan::python::pylib::ScanOperatorHandle;
        use pyo3::prelude::*;
        let Range {
            start,
            end,
            step,
            num_partitions,
        } = range;

        let partitions = num_partitions.unwrap_or(1);

        ensure!(partitions > 0, "num_partitions must be greater than 0");

        let start = start.unwrap_or(0);

        let step = usize::try_from(step).wrap_err("step must be a positive integer")?;
        ensure!(step > 0, "step must be greater than 0");

        let plan = Python::with_gil(|py| {
            let range_module =
                PyModule::import(py, "daft.io._range").wrap_err("Failed to import range module")?;

            let range = range_module
                .getattr(pyo3::intern!(py, "RangeScanOperator"))
                .wrap_err("Failed to get range function")?;

            let range = range
                .call1((start, end, step, partitions))
                .wrap_err("Failed to create range scan operator")?
                .into_pyobject(py)
                .unwrap()
                .unbind();

            let scan_operator_handle = ScanOperatorHandle::from_python_scan_operator(range, py)?;

            let plan = LogicalPlanBuilder::table_scan(scan_operator_handle.into(), None)?;

            eyre::Result::<_>::Ok(plan)
        })
        .wrap_err("Failed to create range scan")?;

        Ok(plan)
    }
}
