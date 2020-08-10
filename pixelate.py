from PIL import Image


def pixelate(filename, new_filename, block_size):
    image = Image.open(filename)
    image = image.convert("L")
    # Calculates the number of blocks in each row of the grid.
    grid_width = image.width // block_size
    # Calculates the number of rows in the grid.
    grid_height = image.height // block_size
    # Initializes the new image.
    new_image = Image.new("L", (image.width, image.height))
    for i in range(grid_height):
        # Calculates the starting y-position of the block.
        y_start = i * block_size
        for j in range(grid_width):
            # Calculates the starting x-position.
            x_start = j * block_size
            # Initializes the total lightness of the block.
            total_lightness = 0
            # Iterates over the x-values.
            for x in range(x_start, x_start + block_size):
                # Iterates over the y-values.
                for y in range(y_start, y_start + block_size):
                    # Adds the value at (x,y) to the total.
                    total_lightness += image.getpixel((x,y))
            number_of_pixels = block_size * block_size
            average_lightness = total_lightness // number_of_pixels
            # Iterates over the x-values again.
            for x in range(x_start, x_start + block_size):
                # Iterates over the y-values again.
                for y in range(y_start, y_start + block_size):
                    # Sets the value of the new image at (x,y).
                    new_image.putpixel((x,y), (average_lightness,))
    # Draws horizontal grid lines.
    for i in range(1, grid_height):
        y_start = i * block_size
        for x in range(new_image.width):
            new_image.putpixel((x, y_start), (0,))
    # Draws vertical grid lines.
    for j in range(1, grid_width):
        x_start = j * block_size
        for y in range(new_image.height):
            new_image.putpixel((x_start, y), (0,))

    new_image.save(new_filename)


pixelate("sample Sigma (150x150).png", "pixelated Sigma (150x150).png", 15)