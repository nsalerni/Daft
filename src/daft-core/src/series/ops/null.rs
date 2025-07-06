use common_error::DaftResult;

use crate::series::Series;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FillNullStrategy {
    Forward,
    Backward,
}

impl Series {
    pub fn is_null(&self) -> DaftResult<Self> {
        self.inner.is_null()
    }

    pub fn not_null(&self) -> DaftResult<Self> {
        self.inner.not_null()
    }

    pub fn fill_null(&self, fill_value: &Self) -> DaftResult<Self> {
        let predicate = self.not_null()?;
        self.if_else(fill_value, &predicate)
    }

    pub fn fill_null_with_strategy(&self, strategy: FillNullStrategy) -> DaftResult<Self> {
        match strategy {
            FillNullStrategy::Forward => self.fill_null_forward(),
            FillNullStrategy::Backward => self.fill_null_backward(),
        }
    }

    fn fill_null_forward(&self) -> DaftResult<Self> {
        let len = self.len();
        if len == 0 {
            return Ok(self.clone());
        }

        let mut result = self.clone();
        let mut last_valid_value: Option<Self> = None;

        for i in 0..len {
            let current_value = self.slice(i, i + 1)?;
            let is_null = current_value.is_null()?.slice(0, 1)?;

            // Check if the scalar value at index 0 is true (indicating null)
            if is_null.bool()?.get(0).unwrap() {
                // Current value is null, use the last valid value if available
                if let Some(ref fill_val) = last_valid_value {
                    let mask = self.create_mask_for_index(i)?;
                    result = result.if_else(fill_val, &mask)?;
                }
            } else {
                // Current value is not null, update last_valid_value
                last_valid_value = Some(current_value);
            }
        }

        Ok(result)
    }

    fn fill_null_backward(&self) -> DaftResult<Self> {
        let len = self.len();
        if len == 0 {
            return Ok(self.clone());
        }

        let mut result = self.clone();
        let mut next_valid_value: Option<Self> = None;

        // Scan backwards to find next valid values
        for i in (0..len).rev() {
            let current_value = self.slice(i, i + 1)?;
            let is_null = current_value.is_null()?.slice(0, 1)?;

            // Check if the scalar value at index 0 is true (indicating null)
            if is_null.bool()?.get(0).unwrap() {
                // Current value is null, use the next valid value if available
                if let Some(ref fill_val) = next_valid_value {
                    let mask = self.create_mask_for_index(i)?;
                    result = result.if_else(fill_val, &mask)?;
                }
            } else {
                // Current value is not null, update next_valid_value
                next_valid_value = Some(current_value);
            }
        }

        Ok(result)
    }

    fn create_mask_for_index(&self, index: usize) -> DaftResult<Self> {
        use arrow2::array::Array;

        // Create a boolean mask that's true only at the specified index
        let mut mask_data = vec![false; self.len()];
        mask_data[index] = true;

        let field = daft_schema::field::Field::new("mask", crate::datatypes::DataType::Boolean);
        Self::from_arrow(
            std::sync::Arc::new(field),
            arrow2::array::BooleanArray::from_slice(mask_data).to_boxed(),
        )
    }
}
