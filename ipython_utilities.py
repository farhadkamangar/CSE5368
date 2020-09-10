try:
    if __IPYTHON__:
        from IPython import get_ipython
        from IPython.display import display, HTML
        from ipywidgets import Layout, HBox, VBox, interact, \
            interactive, fixed, FloatSlider, \
            IntSlider, Label, Checkbox, FloatRangeSlider, Dropdown

        in_ipython_flag = True
    else:
        in_ipython_flag = False
except:
    in_ipython_flag = False
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import IPython.core.display


def arrange_widgets_in_grid(ob_interactive, number_of_col=2, description_width='50%', height=''):
    """
    This function arranges the widgets in a two dimensional grid.
    Farhad Kamangar
    Oct. 2017
    """
    style = {'description_width': description_width, 'handle_color': 'lightblue'}
    layout = Layout(height=height, width='100%', border='1px solid blue', margin='5px')
    hbox_layout = Layout(border='0px solid blue', width='100%')
    # Set the style and layout of each of the widgets
    for k in range(len(ob_interactive.children) - 1):
        ob_interactive.children[k].layout = layout
        ob_interactive.children[k].style = style
    children_list = []
    current_widget_index = 0
    for row in range(math.ceil((len(ob_interactive.children) - 1) / number_of_col)):
        temp_list = []
        for col in range(number_of_col):
            if (current_widget_index) < (len(ob_interactive.children) - 1):
                temp_list.append(ob_interactive.children[current_widget_index])
                current_widget_index += 1
        children_list.append(HBox(temp_list, layout=hbox_layout))
    children_list.append(ob_interactive.children[-1])
    ob_interactive.children = tuple(children_list)
    display(ob_interactive)
    return


def display_as_html_table(input_list_or_array, title='', first_row="", first_column="", cell_format="0.2f", div_id=''):
    """ This function displays a 2d list or numpy array as an HTML table.
    It also allows the user to specify row and column labels (top row and left column).
    Farhad Kamangar June 2017"""
    #     format_string = '<td>{0:' + cell_format + '}</td>'
    style_html = '''<style>
    table {border: solid;border-collapse: collapse;}
    tr {outline: thin solid black;}
    th {background-color:#AAFFFF;outline: thin solid black;}
    td {background-color:#EEFFFF; outline: thin solid black;}
    </style>
    '''
    table_html = '<b>' + title + '</b>' + '<table>'
    # convert the input to a list
    if type(input_list_or_array) == np.ndarray:
        current_array = input_list_or_array.tolist()
    else:
        current_array = input_list_or_array
    # In case number of columns in each row are not equal.
    # Find the maximum number of columns

    if isinstance(current_array[0], list):  # Input is a 2d list
        max_num_of_columns = 0
        for current_row in current_array:
            if len(current_row) > max_num_of_columns:
                max_num_of_columns = len(current_row)
        # Add column labels
        if first_row:
            table_html += "<tr>"
            if first_column:
                table_html += '<th> </th>'
            for k in range(max_num_of_columns):
                try:
                    table_html += '<th><b>{0:s}</b></th>'.format(first_row[k])
                except:
                    table_html += '<th> </th>'
            table_html += "</tr>"
        for row_index, current_row in enumerate(current_array):
            table_html += "<tr>"
            if first_column:
                try:
                    table_html += '<td><b>{0:s}</b></td>'.format(first_column[row_index])
                except:
                    table_html += '<td> </td>'
            for k in range(max_num_of_columns):
                try:
                    if isinstance(current_row[k], int):
                        table_html += ('<td>{0:d}</td>').format(current_row[k])
                    elif isinstance(current_row[k], float):
                        table_html += ('<td>{0:' + cell_format + '}</td>').format(current_row[k])
                    else:
                        table_html += ('<td>{0:s}</td>').format(str(current_row[k]))
                except:
                    table_html += '<td> </td>'
                    print("exception")
            table_html += "</tr>"
    else:  # Input is a 1d list
        if first_row:
            table_html += "<tr>"
            if first_column:
                if first_column:
                    table_html += '<th> </th>'
            for k in range(len(current_array)):
                try:
                    table_html += '<th><b>{0:s}</b></th>'.format(first_row[k])
                except:
                    table_html += '<th> </th>'
            table_html += "</tr>"

        table_html += "<tr>"
        if first_column:
            try:
                table_html += '<td><b>{0:s}</b></td>'.format(first_column[0])
            except:
                table_html += '<td> </td>'
        for k in range(len(current_array)):
            try:
                if isinstance(current_array[k], int):
                    table_html += ('<td>{0:d}</td>').format(current_array[k])
                elif isinstance(current_array[k], float):
                    table_html += ('<td>{0:' + cell_format + '}</td>').format(current_array[k])
                else:
                    table_html += ('<td>{0:s}</td>').format(str(current_array[k]))
            except:
                table_html += '<td> </td>'
                print("exception")
        table_html += "</tr>"

    table_html += "</table>"
    return display_string_as_html_in_div(table_html, div_id=div_id)


def display_string_as_html_in_div(input_string, div_id=""):
    style_html = '''<style>
     table {border: solid;border-collapse: collapse;}
     tr {outline: thin solid black;}
     th {background-color:#AAFFFF;outline: thin solid black;}
     td {background-color:#EEFFFF; outline: thin solid black;}
     </style>'''

    if div_id:
        temp_string = input_string.replace('\n', ' ').replace('\r', ' ').replace('\\', '\\\\')
        html_script = '''<head>
        <script type="text/javascript">
            function myfunction() {
               document.getElementById("''' + div_id + '''").innerHTML="''' + temp_string + '''"; 
            }
        </script>
    </head>
    <body>
    <div style="margin: 2px;padding: 2px;background-color:#AAFFFF;" id="''' + div_id + '''"></div>
        <script type="text/javascript">
            myfunction();
        </script>
    </body>'''

    else:
        html_script = '''<head>''' + style_html + '''</head><body>''' + input_string + '''</body>'''
    display(HTML(html_script))
    return div_id


def display_an_image(input_image, width_inches=2, height_inches=2, title="", ticks=False):
    """ This function displays a single image.
     The input image is assumed to be a numpy array
    Farhad Kamangar Mar. 2017"""
    fig1, axes_array = plt.subplots(1, 1)
    fig1.set_size_inches(width_inches, height_inches)
    fig1.suptitle(title)
    if input_image.ndim == 2:
        axes_array.imshow(input_image, cmap=plt.cm.gray)
    else:
        axes_array.imshow(input_image)
    if not ticks:
        axes_array.axis('off')
    plt.show()
    plt.pause(0.001)


def display_images_on_grid(images, number_of_rows=0, number_of_columns=0,
                           image_height_inches=0, image_width_inches=0,
                           grid_height_inches=0, grid_width_inches=0,
                           title="", ticks=False):
    """ This function displays N images on a grid.
    It is assumed that each image is a numpy array.
    This function does not display a single image
    Farhad Kamangar Mar. 2017"""
    if isinstance(images, np.ndarray):
        number_of_images = images.shape[0]
    else:  # Not a numpy array
        number_of_images = len(images)
    if (grid_width_inches > 0) and (grid_width_inches < image_width_inches):
        # In case of conflict grid width has priority
        image_width_inches = 0
    if (grid_height_inches > 0) and (grid_height_inches < image_height_inches):
        # In case of conflict grid height has priority
        image_height_inches = 0
    if grid_width_inches:
        # Grid width is specified
        if image_width_inches:
            # Grid width and image width are specified. Calculate number of columns
            number_of_columns = np.int(round(grid_width_inches / image_width_inches))
            number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
            number_of_columns_is_forced = True
        else:
            # Grid width is specified but image width is not.
            if number_of_columns:
                # number of columns is specified
                # number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
                number_of_columns_is_forced = True
            else:
                # number of columns is NOT specified
                number_of_columns_is_forced = False


    else:
        # Grid width is NOT specified
        if image_width_inches:
            # Grid width is NOT specifed but image width IS specified.
            if number_of_columns:
                # number of columns IS specified
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
                grid_width_inches = number_of_columns * image_width_inches
                number_of_columns_is_forced = True
            else:
                # number of columns is NOT specified
                grid_width_inches = 4
                number_of_columns_is_forced = False

                # grid_width_inches=number_of_rows*image_width_inches
        else:
            # Grid width and image width are NOT specified.
            grid_width_inches = 4
            if number_of_columns:
                # number of columns is specified
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
                number_of_columns_is_forced = True
            else:
                # number of columns is NOT specified
                number_of_columns_is_forced = False

    if grid_height_inches:
        # Grid height IS specified
        if image_height_inches:
            # Image height IS specified
            if number_of_columns_is_forced:
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
            else:
                number_of_rows = np.int(round(grid_height_inches / image_height_inches))
                number_of_columns = np.int(np.ceil(number_of_images / number_of_rows))
        else:
            # Image height IS NOT specified
            if number_of_columns_is_forced:
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
            else:
                if number_of_rows:
                    # number_of_rows IS specified
                    number_of_columns = np.int(np.ceil(number_of_images / number_of_rows))
                else:
                    number_of_columns = np.int(np.ceil(np.sqrt(number_of_images)))
                    number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
    else:
        # Grid height IS NOT specified
        if image_height_inches:
            # Image height IS specified
            if number_of_columns_is_forced:
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))

            else:
                if number_of_rows:
                    # number_of_rows IS specified
                    number_of_columns = np.int(np.ceil(number_of_images / number_of_rows))
                else:
                    number_of_columns = np.int(np.ceil(np.sqrt(number_of_images)))
                    number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))

            grid_height_inches = number_of_rows * image_height_inches

        else:
            # Image height IS NOT specified
            if number_of_columns_is_forced:
                number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
            else:
                if number_of_rows:
                    # number_of_rows IS specified
                    number_of_columns = np.int(np.ceil(number_of_images / number_of_rows))
                else:
                    number_of_columns = np.int(np.ceil(np.sqrt(number_of_images)))
                    number_of_rows = np.int(np.ceil(number_of_images / number_of_columns))
            grid_height_inches = 2

    if number_of_columns > number_of_images:
        number_of_columns = number_of_images
        number_of_rows = 1
    elif number_of_rows > number_of_images:
        number_of_rows = number_of_images
        number_of_columns = 1

    fig1, axes_array = plt.subplots(number_of_rows, number_of_columns)
    fig1.set_size_inches(grid_width_inches, grid_height_inches)
    fig1.suptitle(title)
    if not ticks:
        plt.axis('off')
    flaten_axes = np.ravel(axes_array)
    for image_index in range(number_of_rows * number_of_columns):
        if image_index >= number_of_images:
            flaten_axes[image_index].axis('off')
            continue
        if images[image_index].ndim == 2:
            flaten_axes[image_index].imshow(images[image_index], cmap=plt.cm.gray)
        else:
            flaten_axes[image_index].imshow(images[image_index])
        if not ticks:
            flaten_axes[image_index].axis('off')
    plt.show()
    plt.pause(0.001)


def read_all_images_in_a_directory(path="", max_number_images_to_read=0, image_type='jpg'):
    # This function reads all the images in a directory and
    # returns a list of numpy arrays consisting of all the images
    if not path:
        path = os.getcwd()
    image_paths = [os.path.join(path, file_name) for
                   file_name in os.listdir(path) if file_name.endswith('.' + image_type)]

    returned_images = []
    file_names = []
    for image_number, image_path in enumerate(image_paths):
        file_names.append(image_path)
        # Read the image
        current_image = Image.open(image_path)
        # Convert the image format into numpy
        current_image = np.array(current_image)

        returned_images.append(current_image)
        if max_number_images_to_read and (image_number >= max_number_images_to_read - 1):
            break
    return (returned_images, file_names)


def display_numpy_array_as_latex(input_array='', before_equal_sign="", title='', number_format="0.2f", div_id=''):
    if len(input_array.shape) > 2:
        raise ValueError('Can not display arrays with more than two dimensions')

    latex_string = title + '<p>' + r'$' + before_equal_sign + r'\left[\begin{array}{*{'
    if len(input_array.shape) == 1:
        # 1d horizontal array
        latex_string += '1}c}\n'
        for x in range(len(input_array)):
            current_value = input_array[x]
            if np.equal(np.mod(current_value, 1), 0):  # if whole number
                latex_string += '{:d}'.format(np.int(current_value))
            else:
                latex_string += ('{:' + number_format + '}').format(current_value)
            latex_string += r' & '
        latex_string = latex_string[:-2]
        latex_string += r'\\'
        latex_string += '\n'
    else:
        latex_string += str(len(input_array[0])) + '}c}\n'
        for x in range(len(input_array)):
            for y in range(len(input_array[x])):
                current_value = input_array[x][y]
                if np.equal(np.mod(current_value, 1), 0):  # if whole number
                    latex_string += '{:d}'.format(np.int(current_value))
                else:
                    latex_string += ('{:' + number_format + '}').format(current_value)
                latex_string += r' & '
            latex_string = latex_string[:-2]
            latex_string += r'\\'
            latex_string += '\n'
    latex_string += r'\end{array}\right]$'
    return display_string_as_html_in_div(latex_string, div_id=div_id)


if __name__ == "__main__":
    def test_widgets(a, b, c, d, e):
        print("a=", a, "b=", b, "c=", c, "d=", d, "e=", e)


    ob_interactive = interactive(test_widgets, a=IntSlider(min=3, max=51, step=2,
                                                           value=17, description='xxxxxxxxxxxxxxxxxxxxxxxxxxxx'),
                                 b=FloatSlider(min=3, max=51, step=2,
                                               value=19, description='zzzzzzzzzzzzz'),
                                 c=Checkbox(description="this is a check box"),
                                 d=Label(value="This is a label"),
                                 e=Dropdown(options=["Option 1", "Option 2", "Option 3"],
                                            value="Option 1",
                                            description="DropDown", continuous_update=False))
    if in_ipython_flag:
        print("Testing arrange_widgets_in_grid")
        arrange_widgets_in_grid(ob_interactive, number_of_col=2)
        print("Testing display_as_html_table with 2-d list with headers")
        display_as_html_table([[3000, 2, 6], [1, 4, 5]], "With Headers",
                              first_row=['First', 'Second', 'Third'], first_column=['One', 'Two'])
        print("Testing display_as_html_table with 1-d list no headers")
        display_as_html_table([3000, 2, 6], "No Headers")

        print("Testing display_as_html_table with 1-d list with headers")
        display_as_html_table([3000, "Hello", 6], "With Headers",
                              first_row=['First', 'Second', 'Third'], first_column=['One', 'Two'])

        print("Testing display_as_html_table to see if output of the next table can overwrite")
        print(" this table (Because the div_id of both tables are the same)")
        div_id = display_as_html_table([[3, 5, 6], [1, 4, 5]], "Header of the first table",
                                       first_row=['First', 'Second', 'Third'], first_column=['One', 'Two'], div_id="t1")

        display_as_html_table([['This is the second table', 'b', 6.5365], [1, 4, 5]], "Header of the second table",
                              first_row=['First', 'Second', 'Third'], first_column=['One', 'Two'], div_id=div_id)
        d = r'''This is an inline equation $x_{abcd}= \sqrt {a^2 + b^2} + \alpha$'''
        e = r'''$$S{H_{z_{xz}}} \alpha = \left[ {\matrix{ 1 & 0 & 0 & 0  \cr 
   0 & 1 & 0 & 0  \cr 
   0 & c & 1 & 0  \cr 
   0 & 0 & 0 & 1  \cr 
 } } \right]$$ '''

        print("testing display_string_as_html_in_div no div")
        div_id = display_string_as_html_in_div(d)
        display_string_as_html_in_div(e, div_id=div_id)
        print("testing display_string_as_html_in_div with div")
        div_id = display_string_as_html_in_div(d, div_id='test1')
        display_string_as_html_in_div(e, div_id=div_id)
        A = np.array([[12, 5, 2.34567],
                      [20, 4, 8],
                      [2, 4, 3],
                      [7, 1, 10]])
        B = np.array([[1], [2], [4]])
        C = np.array([1, 2, 3])
        print("Testing display_numpy_array_as_latex no div")
        display_numpy_array_as_latex(A, before_equal_sign='A=', title="A Title")
        display_numpy_array_as_latex(B, before_equal_sign='B=', title="B Title")
        display_numpy_array_as_latex(C, before_equal_sign='C=', title="C Title")

        print("Testing display_numpy_array_as_latex. show the result in a given div")
        div_id = display_numpy_array_as_latex(A, before_equal_sign='A=', title="A Title", div_id='latex_div')
        display_numpy_array_as_latex(B, before_equal_sign='B=', title="B Title", div_id=div_id)
        display_numpy_array_as_latex(C, before_equal_sign='C=', title="C Title", div_id=div_id)

    print('Testing display_images_on_grid')
    images, file_names = read_all_images_in_a_directory("",
                                                        max_number_images_to_read=20, image_type='jpg')
    display_images_on_grid(images)
    print('Testing display_images_on_grid')
    display_images_on_grid(images, number_of_columns=2,
                           number_of_rows=50, grid_height_inches=10,
                           grid_width_inches=10)
    images = images[0]
    display_an_image(images)



# This part was developed based on Eric Wieser's post
def _html_repr_helper(contents, index, is_horz):
    dims_left = contents.ndim - len(index)
    if dims_left == 0:
        s = contents[index]
    else:
        s = '<span class="numpy-array-comma">,</span>'.join(
            _html_repr_helper(contents, index + (i,), is_horz) for i in range(contents.shape[len(index)])
        )
        s = ('<span class="numpy-array-bracket numpy-array-bracket-open">[</span>'
             '{}'
             '<span class="numpy-array-bracket numpy-array-bracket-close">]</span>'.format(s))

    # apply some classes for styling
    classes = []
    classes.append('numpy-array-slice')
    classes.append('numpy-array-ndim-{}'.format(len(index)))
    classes.append('numpy-array-ndim-m{}'.format(dims_left))
    if is_horz(contents, len(index)):
        classes.append('numpy-array-horizontal')
    else:
        classes.append('numpy-array-vertical')

    hover_text = '[{}]'.format(','.join('{}'.format(i) for i in (index + (':',) * dims_left)))

    return "<span class='{}' title='{}'>{}</span>".format(
        ' '.join(classes), hover_text, s,
    )


basic_css = """
    .numpy-array {
        display: inline-block;
    }
    .numpy-array .numpy-array-slice {
        border: 1px solid #cfcfcf;
        border-radius: 4px;
        margin: 1px;
        padding: 1px;
        display: flex;
        flex: 1;
        text-align: right;
        position: relative;
    }
    .numpy-array .numpy-array-slice:hover {
        border: 1px solid #66BB6A;
    }
    .numpy-array .numpy-array-slice.numpy-array-vertical {
        flex-direction: column;
    }
    .numpy-array .numpy-array-slice.numpy-array-horizontal {
        flex-direction: row;
    }
    .numpy-array .numpy-array-ndim-m0 {
        padding: 0 0.5ex;
    }

    /* Hide the comma and square bracket characters which exist to help with copy paste */
    .numpy-array .numpy-array-bracket {
        font-size: 0;
        position: absolute;
    }
    .numpy-array span .numpy-array-comma {
        font-size: 0;
        height: 0;
    }
"""

show_brackets_css = """
    .numpy-array.show-brackets .numpy-array-slice {
        border-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-bracket {
        border: 1px solid black; 
        border-radius: 0;  /* looks better without... */
    }
    .numpy-array.show-brackets .numpy-array-horizontal > .numpy-array-bracket-open {
        top: -1px;
        bottom: -1px;
        left: -1px;
        width: 10px;
        border-right: none;
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-horizontal > .numpy-array-bracket-close {
        top: -1px;
        bottom: -1px;
        right: -1px;
        width: 10px;
        border-left: none;
        border-top-left-radius: 0;
        border-bottom-left-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-vertical > .numpy-array-bracket-open {
        top: -1px;
        right: -1px;
        left: -1px;
        height: 10px;
        border-bottom: none;
        border-bottom-right-radius: 0;
        border-bottom-left-radius: 0;
    }
    .numpy-array.show-brackets .numpy-array-vertical > .numpy-array-bracket-close {
        left: -1px;
        bottom: -1px;
        right: -1px;
        height: 10px;
        border-top: none;
        border-top-right-radius: 0;
        border-top-left-radius: 0;
    }
"""


def print_html(x, show_brackets=False, is_horz=lambda arr, ax: ax == arr.ndim - 1):
    classes = ['numpy-array']
    css = basic_css
    if show_brackets:
        classes += ['show-brackets']
        css += show_brackets_css
    return IPython.core.display.HTML(
        """<style>{}</style><div class='{}'>{}</div>""".format(
            css,
            ' '.join(classes),
            _html_repr_helper(x, (), is_horz))
    )