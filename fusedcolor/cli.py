import fusedcolor.base as b

def main(f:str,input_image_path: str, output_directory: str):  # pragma: no cover
    match f:
        case "save_cmyk_channels":
            b.save_cmyk_channels(input_image_path, output_directory)
        case "make_stencils":
            b.make_stencils(input_image_path,output_directory)
        case "showLines":
            b.showLines(input_image_path,output_directory)