class Image3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, target_depth=64, target_size=(128,128)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.target_size = target_size
        self.target_depth = target_depth

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def standardize_pixel_array(self, dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    # Correct DICOM pixel_array if PixelRepresentation == 1.
        pixel_array = dcm.pixel_array
        if dcm.PixelRepresentation == 1:
            bit_shift = dcm.BitsAllocated - dcm.BitsStored
            dtype = pixel_array.dtype 
            pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift
        return pixel_array

    
    def resize_img(self, img_paths):
        volume_shape = (self.target_size[0], self.target_size[1], len(img_paths)) 
        volume = np.zeros(volume_shape, dtype=np.float64)
        for i, image_path in enumerate(img_paths): 
            image = pydicom.read_file(image_path)
            image = self.standardize_pixel_array(image)
            image = cv2.resize(image, self.target_size)
            volume[:,:,i] = image
        return volume
    
    def change_depth_siz(self, patient_volume):
        current_depth = patient_volume.shape[-1]
        depth = current_depth / self.target_depth
        depth_factor = 1 / depth
        img_new = zoom(patient_volume, (1, 1, depth_factor), mode='nearest')
        return img_new
    
    def normalize_volume(self, resized_volume):
        original_shape = resized_volume.shape
        flattened_image = resized_volume.reshape((-1,))
        scaler = preprocessing.MinMaxScaler()
        normalized_flattened_image = scaler.fit_transform(flattened_image.reshape((-1, 1)))
        normalized_volume_image = normalized_flattened_image.reshape(original_shape)
        return normalized_volume_image
    
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        resized_shape = (len(batch_x), self.target_size[0], self.target_size[1], self.target_depth)
        resized_images = np.zeros(resized_shape, dtype=np.float64)
        for i, list_files in enumerate(batch_x):
            preprocessed_images = self.resize_img(list_files)
            resized_images_siz = self.change_depth_siz(preprocessed_images)
            normalized_volume = self.normalize_volume(resized_images_siz)
            resized_images[i,:,:,:] = normalized_volume
        return resized_images, np.array(batch_y)