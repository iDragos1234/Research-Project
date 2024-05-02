from preprocessor import Preprocessor

def main():
    print('Start preprocessing ...')

    image_file_base = 'OAI-9003175-V06-20090723'

    preprocessor = Preprocessor(image_file_base)
    preprocessor\
        .load_data()\
        .load_points()\
        .get_source_pixel_spacing()\
        .resample_to_target_resolution()\
        .check_photometric_interpretation()\
        .check_VOILUT_function()\
        .plot_image()\
        .plot_curves()
    
    print(preprocessor.pixel_spacing)

if __name__ == '__main__':
    main()