import coremltools

output_labels = ['Benign', 'Malignant']

coreml_model = coremltools.converters.keras.convert(
    'Models/ft_vl_vgg16.h5',
    input_names='image',
    image_input_names='image',
    output_names='output',
    class_labels= output_labels,
    image_scale=1
)

coreml_model.save('Models/FTVGG16.mlmodel')
