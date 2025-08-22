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

"""HTML Vis."""
import os
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
# pylint: disable=g-complex-comprehension
# pylint: disable=using-constant-test
# pylint: disable=g-explicit-length-test
# pylint: disable=g-importing-member
import os.path as osp

import imageio


class HTMLVisBase:

  def __init__(self, save_dir, key_order, per_page=5):
    self.header = None
    self.key_order = key_order
    self.results = []
    self.per_page = per_page
    self.save_dir = save_dir

  @staticmethod
  def start_basic_html(fp, num_pages, curr_pageno):
    fp.write("<html>")
    ## paging stuff
    fp.write("<body>")
    fp.write("<p> Paging </p>")
    fp.write("<table> <tr> ")
    for i in range(num_pages):
      if i == curr_pageno:
        fp.write(f"<td>{i}</td>")
      else:
        fp.write(f'<td> <a href="./view_{i}.html"> {i} </a></td>')
    fp.write("</table> </tr>")
    return

  @staticmethod
  def end_basic_html(fp):
    fp.write("</body>")
    fp.write("</html>")
    return

  def write_html(self,):

    save_dir = self.save_dir
    results = self.results

    per_page = self.per_page
    if per_page > 0:
      num_pages = len(results) // per_page + 1
    else:
      num_pages = 1
      per_page = len(results)

    for pageno in range(num_pages):
      fp = open(osp.join(save_dir, f"view_{pageno}.html"), "w")
      page_items = results[pageno * per_page:per_page * (pageno + 1)]
      self.start_basic_html(fp, num_pages=num_pages, curr_pageno=pageno)
      self.write_results_block(fp, self.key_order, page_items, save_dir)
      self.end_basic_html(fp)
      self.close_basic_html(fp)
    return

  @staticmethod
  def write_results_block(fp, keyorder, results, save_dir):
    raise NotImplementedError

  def add_results(self, results):
    self.results.append(results)

  @staticmethod
  def close_basic_html(fp):
    fp.close()


class HTMLVis(HTMLVisBase):

  @staticmethod
  def write_results_block(fp, keyorder, results, save_dir):

    fp.write("<table>\n")
    fp.write("<tr>")
    for keyname in keyorder:
      fp.write(f"<th> {keyname} </th>")
    fp.write("</tr>")

    for result in results:
      fp.write("<tr>\n")
      index = result["index"]
      index_save_dir = osp.join(save_dir, "media", f"{index}")
      os.makedirs(index_save_dir, exist_ok=True)
      for key in keyorder:
        if key == "index":
          fp.write(f"<td> {result[key]} </td>")
        else:
          img_name = f"{key}.png"
          img_path = osp.join(index_save_dir, img_name)
          imageio.imsave(img_path, result[key])
          fp.write(f'<td> <img src="./media/{index}/{img_name}" /> </td>')
      fp.write("</tr>\n")
    fp.write("</table>\n")
    return
