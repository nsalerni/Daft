use common_error::DaftResult;

use crate::{
    array::{growable::make_growable, StructArray},
    datatypes::Field,
    DataType, IntoSeries, Series,
};

use super::{bitmap_growable::ArrowBitmapGrowable, Growable};

pub struct StructGrowable<'a> {
    name: String,
    dtype: DataType,
    children_growables: Vec<Box<dyn Growable + 'a>>,
    growable_validity: ArrowBitmapGrowable<'a>,
}

impl<'a> StructGrowable<'a> {
    pub fn new(
        name: &str,
        dtype: &DataType,
        arrays: Vec<&'a StructArray>,
        use_validity: bool,
        capacity: usize,
    ) -> Self {
        match dtype {
            DataType::Struct(fields) => {
                let children_growables: Vec<Box<dyn Growable>> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        make_growable(
                            f.name.as_str(),
                            &f.dtype,
                            arrays
                                .iter()
                                .map(|a| a.children.get(i).unwrap())
                                .collect::<Vec<_>>(),
                            use_validity,
                            capacity,
                        )
                    })
                    .collect::<Vec<_>>();
                let growable_validity = ArrowBitmapGrowable::new(
                    arrays.iter().map(|a| a.validity()).collect(),
                    capacity,
                );
                Self {
                    name: name.to_string(),
                    dtype: dtype.clone(),
                    children_growables,
                    growable_validity,
                }
            }
            _ => panic!("Cannot create StructGrowable from dtype: {}", dtype),
        }
    }
}

impl<'a> Growable for StructGrowable<'a> {
    fn extend(&mut self, index: usize, start: usize, len: usize) {
        for child_growable in &mut self.children_growables {
            child_growable.extend(index, start, len)
        }
        self.growable_validity.extend(index, start, len);
    }

    fn add_nulls(&mut self, additional: usize) {
        for child_growable in &mut self.children_growables {
            child_growable.add_nulls(additional);
        }
        self.growable_validity.add_nulls(additional);
    }

    fn build(&mut self) -> DaftResult<Series> {
        let grown_validity = std::mem::take(&mut self.growable_validity);

        let built_children = self
            .children_growables
            .iter_mut()
            .map(|cg| cg.build())
            .collect::<DaftResult<Vec<_>>>()?;
        let built_validity = grown_validity.build();
        Ok(StructArray::new(
            Field::new(self.name.clone(), self.dtype.clone()),
            built_children,
            Some(built_validity),
        )
        .into_series())
    }
}
