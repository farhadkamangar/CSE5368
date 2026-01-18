import numpy as np
import IPython.core.display
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