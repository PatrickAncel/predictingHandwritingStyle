from PIL import Image
from os import listdir,path

class ImageMatrix:
    def __init__(self, image = None, filepath = ""):
        if image == None:
            return
        # Stores the filepath of the image.
        self.filepath = filepath
        # Stores the dimensions of the image.
        self.width = image.width
        self.height = image.height
        pixel_count = self.width * self.height
        # Initializes the pixel array.
        self.pixel = [0 for i in range(pixel_count)]
        # Converts the image to grayscale.
        grayscale_image = image.convert('L')
        # Iterates over the rows.
        pixels_added = 0
        for y in range(self.height):
            # Iterates over the columns.
            for x in range(self.width):
                # Adds the pixel at (x,y) to the pixel array.
                self.pixel[pixels_added] = grayscale_image.getpixel((x, y))
                pixels_added += 1
    
    def get_pixel(self, x, y):
        # Computes the number of pixels in the rows above y.
        pixels_above = y * self.width
        # The index of the pixel is the number of pixels before it.
        # The pixels above and to the left of this pixel are the ones that come before it.
        pixels_before = pixels_above + x
        return self.pixel[pixels_before]
    
    def to_string(self):
        '''Represents the matrix as "filepath:width:height:rle_compressed_pixel_sequence".'''
        # Initializes the list of (pixel,quantity) tuples.
        runs = []
        # Gets the first pixel.
        pixel_value = self.pixel[0]
        # Initializes the first run.
        current_run = [pixel_value, 0]
        for pixel_value in self.pixel:
            # If this pixel belongs to the current run...
            if pixel_value == current_run[0]:
                # ...increment the run length
                current_run[1] += 1
            else:
                # Stores the run as a tuple.
                run_as_tuple = (current_run[0], current_run[1])
                runs.append(run_as_tuple)
                # Starts a new run.
                current_run = [pixel_value, 1]
        # Stores the final run as a tuple.
        run_as_tuple = (current_run[0], current_run[1])
        runs.append(run_as_tuple)
        # Initializes the list of strings that represent the runs.
        run_strings = []
        # Iterates over the run tuples.
        for (pixel_value, length) in runs:
            if length > 1:
                # Runs of 255 occurs frequently enough to get their own symbol.
                if pixel_value == 255:
                    run_strings.append(F"!{length}")
                # Runs of 0 also get their own symbol.
                elif pixel_value == 0:
                    run_strings.append(F"@{length}")
                else:
                    # Stores the run as "value|length".
                    run_strings.append(F"{pixel_value}|{length}")
            else:
                # Stores the run as just the value.
                run_strings.append(str(pixel_value))
        # Joins the run strings to get the full sequence.
        rle_compressed_pixel_sequence = ",".join(run_strings)
        uncompressed_pixel_sequence = ','.join( [str(p) for p in self.pixel] )
        print(F"Compression Ratio: {len(rle_compressed_pixel_sequence) / len(uncompressed_pixel_sequence)}")
        return F'{self.filepath}:{self.width}:{self.height}:{rle_compressed_pixel_sequence}'
    
    @staticmethod
    def from_string(image_string):
        # Initializes an empty ImageMatrix.
        matrix = ImageMatrix()
        # Splits the string to get the dimensions and pixel sequence.
        (filepath, width, height, rle_compressed_pixel_sequence) = image_string.split(":")
        matrix.filepath = filepath
        matrix.width = int(width)
        matrix.height = int(height)
        # Uses the dimensions to create the pixel list.
        pixel_count = matrix.width * matrix.height
        matrix.pixel = [0 for i in range(pixel_count)]
        # Splits the pixel sequence into run strings.
        run_strings = rle_compressed_pixel_sequence.split(",")
        pixels_set = 0
        # Iterates over the runs.
        for run_string in run_strings:
            value = None
            length = None
            # Checks if this is a 255-run with length > 1.
            if run_string[0] == "!":
                value = 255
                # Gets everything after the "!" and parses to int.
                length = int(run_string[1:])
            # Checks if this is a 0-run with length > 1.
            elif run_string[0] == "@":
                value = 0
                # Gets everything after the "@" and parses to int.
                length = int(run_string[1:])
            # Checks if this run has length > 1
            elif run_string.find("|") != -1:
                # Splits the value and length.
                (value_string, length_string) = run_string.split("|")
                # Parses the value and length strings.
                value = int(value_string)
                length = int(length_string)
            else:
                # The run has no listed length, so it is implied as 1.
                value = int(run_string)
                length = 1
            # Sets a value for every pixel in the run.
            for _i in range(length):
                matrix.pixel[pixels_set] = value
                pixels_set += 1
        return matrix
    
    def __eq__(self, other):
        # Tests for equality of dimensions.
        if self.width != other.width:
            return False
        if self.height != other.height:
            return False
        pixel_count = self.width * self.height
        # Tests for equality of all pixels.
        for i in range(pixel_count):
            if self.pixel[i] != other.pixel[i]:
                return False
        return True
    
    @staticmethod
    def from_filename(filename):
        image = Image.open(filename)
        return ImageMatrix(image, filename)

    def get_lightness_of_square_section(self, block_size, x_start, y_start):
        '''Gets the mean lightness of a block of the matrix.
        
        The block is a square (width: block_size) region of the
        matrix starting at the point (x_start, y_start). The
        function returns the mean value of all the pixels in
        the block.
        '''
        # Tests if the matrix is not square.
        # if self.width != self.height:
        #     raise ValueError("ImageMatrix must be square")
        # Tests if the matrix width is not divisible by the block_size.
        if self.width % block_size != 0:
            raise ValueError(F"ImageMatrix width is not divisible by {block_size}")
        # Tests if the matrix height is not divisible by the block_size.
        if self.height % block_size != 0:
            raise ValueError(F"ImageMatrix height is not divisible by {block_size}")
        # Tests if (x_start, y_start) does not lie on a grid corner.
        if x_start % block_size != 0 or y_start % block_size != 0:
            raise ValueError(F"Point ({x_start}, {y_start}) is not a grid corner.")
        # Tests if (x_start, y_start) is out of bounds.
        if x_start < 0 or x_start >= self.width or y_start < 0 or y_start >= self.height:
            raise ValueError(F"Point ({x_start}, {y_start}) is out of bounds.")
        # Initializes the total lightness.
        total_lightness = 0
        # Iterates over the x-values.
        for x in range(x_start, x_start + block_size):
            # Iterates over the y-values.
            for y in range(y_start, y_start + block_size):
                # Adds the value at (x, y) to the total.
                total_lightness += self.get_pixel(x, y)
        number_of_pixels = block_size * block_size
        average_lightness = total_lightness / number_of_pixels
        return average_lightness
    
    def to_vector(self, block_size):
        '''Compresses the matrix to a simpler vector.
        
        The value of each component is equal to the mean
        lightness of a square section of pixels, divided
        by 255.
        '''
        # Calculates the number of blocks in each row of the grid.
        grid_width = self.width // block_size
        # Calculates the number of rows in the grid.
        grid_height = self.height // block_size
        # Initializes the vector.
        vector = [0 for i in range(grid_width * grid_height)]
        components_set = 0
        for i in range(grid_height):
            # Calculates the starting y-position of the block.
            y_start = i * block_size
            for j in range(grid_width):
                # Calculates the starting x-position.
                x_start = j * block_size
                # Gets the average lightness of the block.
                lightness = self.get_lightness_of_square_section(block_size, x_start, y_start)
                # Makes sure the value is between 0 and 1, inclusive.
                block_value = lightness / 255.0
                # Sets the value of a component.
                vector[components_set] = block_value
                components_set += 1
        return vector

class ImageMatrixList:
    def __init__(self):
        self.matrices = []

    def empty(self):
        '''Removes all of this ImageMatrixList's matrices.'''
        self.matrices = []

    def regulate_dimensions(self, width, height):
        '''Removes any image matrices with incorrect dimensions.'''
        self.matrices = [matrix for matrix in self.matrices if matrix.width == width and matrix.height == height]

    def get_from_image_file(self, filepath):
        '''Processes a single PNG or JPEG file and stores it as an image matrix.'''
        if filepath[-4:].lower() in [".png", ".jpg", "jpeg"]:
            # Reads and processes the image.
            matrix = ImageMatrix.from_filename(filepath)
            # Adds the matrix to the list.
            self.matrices.append(matrix)
            print("Image matrices added: 1")
        else:
            print(F'File was not PNG or JPEG: {filepath}')

    def get_from_folder(self, directory):
        '''Processes all the PNG and JPEG files in a directory and stores them as image matrices.'''
        # Gets everything in the directory.
        files = listdir(directory)
        matrices_added = 0
        # Iterates over the files.
        for filename in files:
            # Detects if this file is a PNG or JPEG.
            if filename[-4:].lower() in [".png", ".jpg", "jpeg"]:
                # Reads and processes the image.
                matrix = ImageMatrix.from_filename(F'{directory}/{filename}')
                # Adds the matrix to the list.
                self.matrices.append(matrix)
                matrices_added += 1
        print(F'Image matrices added: {matrices_added}')
    
    def get_from_text_file(self, filepath):
        '''Creates and adds image matrices using data from a text file.'''
        # Opens the file for reading.
        f = open(filepath, "r")
        # Reads the data from the file.
        data = f.read()
        # Closes the file.
        f.close()
        # Splits the data into separate image matrix strings.
        image_matrix_strings = data.split(";")
        # Iterates over the strings.
        for image_matrix_string in image_matrix_strings:
            # Parses the string to an ImageMatrix.
            matrix = ImageMatrix.from_string(image_matrix_string)
            # Adds the matrix to the list.
            self.matrices.append(matrix)
        print(F'Image matrices added: {len(image_matrix_strings)}')
    
    def store_in_text_file(self, filepath):
        # Will not write over an existing file.
        if path.exists(filepath):
            raise FileExistsError("File already exists!")
        else:
            # Opens the file for writing.
            f = open(filepath, "w")
            try:
                # Converts each ImageMatrix to a string.
                image_matrix_strings = [matrix.to_string() for matrix in self.matrices]
                # Joins the strings into a single string.
                data = ";".join(image_matrix_strings)
                # Writes the data to the file.
                f.write(data)
                # Closes the file.
                f.close()
                # Prints a success message.
                print(F'\nImage matrices written to file {filepath}: {len(self.matrices)}')
            except:
                f.close()
                # Prints a failure message.
                print("\nFailed to encode or store matrix data.")

def process_all_images(folder_names, image_filepaths = [], text_filepaths = [], image_dimensions = (155, 135)):
    '''Creates an ImageMatrixList and obtains data from all specified sources.'''
    matrix_list = ImageMatrixList()
    # Iterates over the folder names.
    for folder_name in folder_names:
        # Gets data from the folder.
        matrix_list.get_from_folder(folder_name)
    # Iterates over the image filepaths.
    for filepath in image_filepaths:
        # Gets data from the image.
        matrix_list.get_from_image_file(filepath)
    # Iterates over the text filepaths.
    for filepath in text_filepaths:
        # Gets data from the text file.
        matrix_list.get_from_text_file(filepath)
    # Removes matrices with incorrect dimensions.
    matrix_list.regulate_dimensions(*image_dimensions)
    return matrix_list