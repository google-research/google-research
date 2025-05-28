# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=g-multiple-import,redefined-builtin
"""show.py.

An abbreviated way to output simple HTML layout of text and images
into a python notebook.

- show a PIL image to show an inline HTML <img>.
- show an array of items to vertically stack them, centered in a block.
- show an array of arrays to horizontally lay them out as inline blocks.
- show an array of tuples to create a table.
From David Bau.
"""
import html as html_module

from IPython.display import display
import PIL.Image, base64, io, IPython, types, sys

g_buffer = None


def blocks(obj, space=''):
  return IPython.display.HTML(space.join(blocks_tags(obj)))


def rows(obj, space=''):
  return IPython.display.HTML(space.join(rows_tags(obj)))


def rows_tags(obj):
  """add row html iterms."""
  if isinstance(obj, dict):
    obj = obj.items()
  results = []
  results.append('<table style="display:inline-table">')
  for row in obj:
    results.append('<tr style="padding:0">')
    for item in row:
      results.append(
          '<td style="text-align:left; vertical-align:top;' + 'padding:1px">'
      )
      results.extend(blocks_tags(item))
      results.append('</td>')
    results.append('</tr>')
  results.append('</table>')
  return results


def blocks_tags(obj):
  """add items in html block."""
  results = []
  if hasattr(obj, '_repr_html_'):
    results.append(obj._repr_html_())  # pylint: disable=protected-access
  elif isinstance(obj, PIL.Image.Image):
    results.append(pil_to_html(obj))
  elif isinstance(obj, (str, int, float)):
    results.append('<div>')
    results.append(html_module.escape(str(obj)))
    results.append('</div>')
  elif isinstance(obj, dict):
    results.extend(blocks_tags([(k, v) for k, v in obj.items()]))
  elif hasattr(obj, '__iter__'):
    if hasattr(obj, 'tolist'):
      # Handle numpy/pytorch tensors as lists.
      try:
        obj = obj.tolist()
      except:  # pylint: disable=bare-except
        pass
    blockstart, blockend, tstart, tend, rstart, rend, cstart, cend = [
        '<div style="display:inline-block;text-align:center;line-height:1;'
        + 'vertical-align:top;padding:1px">',
        '</div>',
        '<table style="display:inline-table">',
        '</table>',
        '<tr style="padding:0">',
        '</tr>',
        '<td style="text-align:left; vertical-align:top; padding:1px">',
        '</td>',
    ]
    needs_end = False
    table_mode = False
    for i, line in enumerate(obj):
      if i == 0:
        needs_end = True
        if isinstance(line, tuple):
          table_mode = True
          results.append(tstart)
        else:
          results.append(blockstart)
      if table_mode:
        results.append(rstart)
        if not isinstance(line, str) and hasattr(line, '__iter__'):
          for cell in line:
            results.append(cstart)
            results.extend(blocks_tags(cell))
            results.append(cend)
        else:
          results.append(cstart)
          results.extend(blocks_tags(line))
          results.append(cend)
        results.append(rend)
      else:
        results.extend(blocks_tags(line))
    if needs_end:
      results.append(table_mode and tend or blockend)
  return results


def pil_to_b64(img, format='png'):
  buffered = io.BytesIO()
  img.save(buffered, format=format)
  return base64.b64encode(buffered.getvalue()).decode('utf-8')


def pil_to_url(img, format='png'):
  return 'data:image/%s;base64,%s' % (format, pil_to_b64(img, format))


def pil_to_html(img, margin=1):
  mattr = ' style="margin:%dpx"' % margin
  return '<img src="%s"%s>' % (pil_to_url(img), mattr)


def a(x, cols=None):
  global g_buffer
  if g_buffer is None:
    g_buffer = []
  g_buffer.append(x)
  if cols is not None and len(g_buffer) >= cols:
    flush()


def reset():
  global g_buffer
  g_buffer = None


def flush(*args, **kwargs):
  global g_buffer
  if g_buffer is not None:
    x = g_buffer
    g_buffer = None
    display(blocks(x, *args, **kwargs))


def show(x=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
  flush(*args, **kwargs)
  if x is not None:
    display(blocks(x, *args, **kwargs))


def html(obj, space=''):
  return blocks(obj, space)._repr_html_()  # pylint:disable=protected-access


class CallableModule(types.ModuleType):

  def __init__(self):
    # or super().__init__(__name__) for Python 3
    types.ModuleType.__init__(self, __name__)
    self.__dict__.update(sys.modules[__name__].__dict__)

  def __call__(self, x=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    show(x, *args, **kwargs)


sys.modules[__name__] = CallableModule()
