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

"""Tests conversion TFGNN GraphTensor to GraphStruct with {tf, jax, np} engines.
"""

import jax
import tensorflow as tf
import tensorflow_gnn

from sparse_deferred import tfgnn as sdtfgnn
import sparse_deferred.jax as sdjax
import sparse_deferred.np as sdnp


_STUDENTS = [
    b'Sami',
    b'Bryan',
    b'Jonathan',
]
_STUDENT_IDX = {value: i for i, value in enumerate(_STUDENTS)}

_COURSES = [
    (b'Arabic', 101),
    (b'ML', 102),
    (b'English', 103),
    (b'Calculus', 104),
]
_COURSE_IDX = {value[0]: i for i, value in enumerate(_COURSES)}

_TOPICS = [
    b'NaturalLanguage',
    b'Math',
]
_TOPIC_IDX = {value: i for i, value in enumerate(_TOPICS)}

_COURSE_TOPICS = [
    (b'Arabic', b'NaturalLanguage'),
    (b'English', b'NaturalLanguage'),
    (b'ML', b'Math'),
    (b'Calculus', b'Math'),
]

_ENROLL_GRADES = {
    (b'Sami', b'Arabic'): 95,
    (b'Bryan', b'English'): 100,
    (b'Sami', b'ML'): 97,
    (b'Bryan', b'Calculus'): 99,
    (b'Bryan', b'ML'): 100,
    (b'Jonathan', b'Calculus'): 100,
    (b'Jonathan', b'Arabic'): 100,
}


def _make_tf_example():
  example = tf.train.Example()

  #### NODES

  # Add student nodes.
  example.features.feature['nodes/students.name'].bytes_list.value.extend(
      _STUDENTS)
  example.features.feature['nodes/students.#size'].int64_list.value.append(
      len(_STUDENTS))

  # Add course nodes.
  course_names, course_codes = zip(*_COURSES)
  example.features.feature['nodes/courses.name'].bytes_list.value.extend(
      course_names)
  example.features.feature['nodes/courses.code'].int64_list.value.extend(
      course_codes)
  example.features.feature['nodes/courses.#size'].int64_list.value.append(
      len(course_names))

  # Add topic nodes.
  example.features.feature['nodes/topics.name'].bytes_list.value.extend(
      _TOPICS)
  example.features.feature['nodes/topics.#size'].int64_list.value.append(
      len(_TOPICS))

  #### EDGES
  example.features.feature['edges/has_topic.#source'].int64_list.value.extend(
      [_COURSE_IDX[course] for (course, _) in _COURSE_TOPICS])
  example.features.feature['edges/has_topic.#target'].int64_list.value.extend(
      [_TOPIC_IDX[topic] for (_, topic) in _COURSE_TOPICS])
  example.features.feature['edges/has_topic.#size'].int64_list.value.append(
      len(_COURSE_TOPICS))

  enrollments, grades = zip(*_ENROLL_GRADES.items())
  enrollment_students, enrollment_courses = zip(*enrollments)

  example.features.feature['edges/enrollments.#source'].int64_list.value.extend(
      [_STUDENT_IDX[student] for student in enrollment_students])
  example.features.feature['edges/enrollments.#target'].int64_list.value.extend(
      [_COURSE_IDX[course] for  course in enrollment_courses])
  example.features.feature['edges/enrollments.grade'].float_list.value.extend(
      grades)
  example.features.feature['edges/enrollments.#size'].int64_list.value.append(
      len(enrollment_courses))

  #### CONTEXT
  example.features.feature['context/root_node'].int64_list.value.append(1)

  return example


def _make_schema():
  schema = tensorflow_gnn.GraphSchema()
  schema.node_sets['students'].features['name'].dtype = (
      tf.string.as_datatype_enum)
  schema.node_sets['courses'].features['name'].dtype = (
      tf.string.as_datatype_enum)
  schema.node_sets['courses'].features['code'].dtype = (
      tf.int32.as_datatype_enum)
  schema.node_sets['courses'].features['code'].shape.dim.add().size = 1
  schema.node_sets['topics'].features['name'].dtype = (
      tf.string.as_datatype_enum)

  schema.edge_sets['has_topic'].source = 'courses'
  schema.edge_sets['has_topic'].target = 'topics'

  schema.edge_sets['enrollments'].source = 'students'
  schema.edge_sets['enrollments'].target = 'courses'
  schema.edge_sets['enrollments'].features['grade'].dtype = (
      tf.float32.as_datatype_enum)
  return schema


class _BaseIOTest(tf.test.TestCase):
  """Tests for io.py, when using TensorFlow as a backend."""

  def _assert_correct(self, graph):
    # Assert nodes are correct.
    self.assertAllEqual(graph.nodes['students']['name'], _STUDENTS)
    course_names, course_codes = zip(*_COURSES)
    self.assertAllEqual(graph.nodes['courses']['name'], course_names)
    self.assertAllEqual(
        graph.nodes['courses']['code'],
        # _make_tf_example adds dimension.
        tf.expand_dims(course_codes, -1))
    self.assertAllEqual(graph.nodes['topics']['name'], _TOPICS)

    (src, tgt), features = graph.edges['has_topic']
    self.assertEmpty(features)  # No features for has_topics!
    has_topic_edges = set(zip(tf.gather(course_names, src).numpy(),
                              tf.gather(_TOPICS, tgt).numpy()))
    self.assertSetEqual(has_topic_edges, set(_COURSE_TOPICS))

    (src, tgt), features = graph.edges['enrollments']
    src_student_names = tf.gather(_STUDENTS, src).numpy()
    tgt_course_names = tf.gather(course_names, tgt).numpy()
    enrollment_edges = set(zip(src_student_names, tgt_course_names))
    self.assertSetEqual(enrollment_edges, set(_ENROLL_GRADES.keys()))
    self.assertIn('grade', features)

    for student, course, grade in zip(
        src_student_names, tgt_course_names, features['grade']):
      self.assertAllEqual(grade, _ENROLL_GRADES[(student, course)])


class TensorflowIOTest(_BaseIOTest):

  def test_graph_struct_from_tf_example(self):
    tf_example = _make_tf_example()
    schema = _make_schema()
    graph_struct = sdtfgnn.graph_struct_from_tf_example(tf_example, schema)
    self._assert_correct(graph_struct)

  def test_graph_struct_from_tfgnn_graph_tensor(self):
    tf_example = _make_tf_example()
    schema = _make_schema()
    graph_spec = tensorflow_gnn.create_graph_spec_from_schema_pb(schema)
    graph_tensor = tensorflow_gnn.parse_single_example(
        graph_spec, tf_example.SerializeToString())
    graph_struct = sdtfgnn.graph_struct_from_graph_tensor(graph_tensor)
    self._assert_correct(graph_struct)


class ToExampleTest(tf.test.TestCase):

  def test_graph_struct_to_tf_example(self):
    tf_example = _make_tf_example()
    schema = _make_schema()

    self.assertProtoEquals(
        tf_example,
        sdtfgnn.graph_struct_to_tf_example(
            sdtfgnn.graph_struct_from_tf_example(tf_example, schema)
        ),
    )


class NumpyIOTest(_BaseIOTest):

  def test_graph_struct_from_tf_example(self):
    tf_example = _make_tf_example()
    schema = _make_schema()
    graph_struct = sdtfgnn.graph_struct_from_tf_example(
        tf_example, schema, engine=sdnp.engine)
    self._assert_correct(graph_struct)

  def test_graph_struct_from_tfgnn_graph_tensor(self):
    tf_example = _make_tf_example()
    schema = _make_schema()
    graph_spec = tensorflow_gnn.create_graph_spec_from_schema_pb(schema)
    graph_tensor = tensorflow_gnn.parse_single_example(
        graph_spec, tf_example.SerializeToString())
    graph_struct = sdtfgnn.graph_struct_from_graph_tensor(
        graph_tensor, engine=sdnp.engine)
    self._assert_correct(graph_struct)


class JaxIOTest(tf.test.TestCase):
  """Jax does not support string features, therefore a (simple) modified graph.

  https://github.com/jax-ml/jax/issues/2084
  """

  def test_graph_struct_from_graph_tensor(self):
    graph_tensor = tensorflow_gnn.GraphTensor.from_pieces(
        context=tensorflow_gnn.Context.from_fields(
            features={'x': tf.constant([1])}),
        node_sets={
            'n1': tensorflow_gnn.NodeSet.from_fields(
                features={'f1': tf.constant([2, 3, 4])},
                sizes=tf.constant([3])),
            'n2': tensorflow_gnn.NodeSet.from_fields(
                features={'f2': tf.constant([[-1.0], [-2.0]])},
                sizes=tf.constant([2])),
        },
        edge_sets={
            'e': tensorflow_gnn.EdgeSet.from_fields(
                adjacency=tensorflow_gnn.Adjacency.from_indices(
                    source=('n1', tf.constant([0, 0, 1, 2])),
                    target=('n2', tf.constant([0, 1, 0, 1]))),
                features={'f': tf.constant([1, 2, 3, 4])},
                sizes=tf.constant([4])),
        }
    )
    graph_struct = sdtfgnn.graph_struct_from_graph_tensor(
        graph_tensor, engine=sdjax.engine)
    self.assertAllEqual(graph_struct.schema['e'], ('n1', 'n2'))
    self.assertAllEqual(graph_struct.nodes['n1']['f1'], [2, 3, 4])
    self.assertAllEqual(graph_struct.nodes['n2']['f2'], [[-1.0], [-2.0]])
    edge_endpoints, edge_features = graph_struct.edges['e']
    self.assertAllEqual(edge_features['f'], [1, 2, 3, 4])
    self.assertAllEqual(edge_endpoints[0], [0, 0, 1, 2])
    self.assertAllEqual(edge_endpoints[1], [0, 1, 0, 1])

    self.assertIsInstance(edge_endpoints[0], jax.Array)
    self.assertIsInstance(edge_endpoints[1], jax.Array)
    self.assertIsInstance(edge_features['f'], jax.Array)
    num_processed_features = 0
    for features in graph_struct.nodes.values():
      for feature_value in features.values():
        num_processed_features += 1
        self.assertIsInstance(feature_value, jax.Array)
    self.assertEqual(num_processed_features, 3)  # two node + one graph feats.


if __name__ == '__main__':
  tf.test.main()
