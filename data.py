import SimpleITK as sitk
import numpy as np  

def readMRIVolume(mri_volume_path):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(mri_volume_path)
    #'dicom_files' are the individual slices of an MRI volume
    reader.SetFileNames(dicom_files)
    retrieved_mri_volume = reader.Execute()
    return retrieved_mri_volume


def resample_image(input_volume, out_spacing=[1, 1, 1]):
  # Resample images to uniform voxel spacing
  
    original_spacing = input_volume.GetSpacing()
    original_size = input_volume.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(input_volume.GetDirection())
    resample.SetOutputOrigin(input_volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_volume.GetPixelIDValue())

    return resample.Execute(input_volume)

def resize_image(input_volume):
    # Resize images to fixed spatial resolution in pixels
    num_axial_slices = int(input_volume.GetSize()[-1])
    output_size = [320, 320, num_axial_slices]
    scale = np.divide(input_volume.GetSize(), output_size)
    spacing = np.multiply(input_volume.GetSpacing(), scale)
    transform = sitk.AffineTransform(3)
    resized_volume = sitk.Resample(input_volume, output_size, transform, sitk.sitkLinear, input_volume.GetOrigin(),
                                  spacing, 
    input_volume.GetDirection())
    return resized_volume


def extract_slices(image_volume):
    image_array = sitk.GetArrayFromImage(image_volume)
    image_slices_array = image_array[int(np.shape(image_array)[0]):int(np.shape(image_array)[0]),:,:]
    return image_slices_array

def extract_slices(image_volume):
    image_array = sitk.GetArrayFromImage(image_volume)
    image_slices_array = image_array[int(np.shape(image_array)[0]):int(np.shape(image_array)[0]),:,:]
    return image_slices_array

image_slice = sitk.GetImageFromArray(image_slices_array)
sitk.WriteImage(image_slice, 'image_slice.png')