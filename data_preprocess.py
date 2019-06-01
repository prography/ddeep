from preprocess import preprocesses

def image_preprocess(input_dir):
    input_datadir = input_dir
    output_datadir = './pre_img'

    obj=preprocesses(input_datadir,output_datadir)
    nrof_images_total,nrof_successfully_aligned=obj.collect_data()

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)



