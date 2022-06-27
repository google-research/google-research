# Copyright 2022 The Google Research Authors.
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

#!/bin/bash

## Script to compile and run eeg_viewer

# Exit on fail
set -e

# Common vars
THIRD_PARTY_FOLDER="third_party"

CSS_BUNDLED_FOLDER="eeg_viewer/static/css"
CSS_BUNDLED_FNAME="${CSS_BUNDLED_FOLDER}/css_styles-bundle.css"

JS_COMPILER="third_party/closure-compiler/closure-compiler-v20190121.jar"
JS_COMPILED_APP="eeg_viewer/static/js/compiled_app.js"
JS_COMPILED_APP_LOADER="eeg_viewer/static/js/compiled_app_loader.js"
JS_APP_LOADER="eeg_viewer/jslib/app_loader.js"
JS_TEMP_LICENSE_WRAPPER="eeg_viewer/jslib/license_wrapper.temp.js"

PROTO_COMPILER_FOLDER="third_party/protoc"
PROTO_COMPILER_CMD="${PROTO_COMPILER_FOLDER}/bin/protoc"
PROTO_TIMESTAMP_FNAME="${PROTO_COMPILER_FOLDER}/include/google/protobuf/timestamp.proto"
PROTO_SRC_FOLDER="./protos"
PROTO_PY_COMPILED_FOLDER="./pyprotos"
PROTO_JS_COMPILED_FOLDER="./jsprotos"

JS_COMMENT="//"
PY_COMMENT="#"

PY_ENV_FOLDER="pyenv"
PY_ENV_ACTIVATE_FNAME="env-activate"


########## UTILS ##########
# Assert the existence of a folder
assert_folder() {
  # $1: folder to assert
  if [[ ! -d "$1" ]]; then
    mkdir -p "$1"
  fi
}


# Ensure that a file is removed
assert_rm() {
  if [[ -f "$1" ]]; then
    rm "$1"
  fi
}


# Return a license header with a given comment char
get_license_header() {
  # $1: comment char
  comment="$1"
  echo "${comment} Copyright 2019 The Google Research Authors.
${comment}
${comment} Licensed under the Apache License, Version 2.0 (the \"License\");
${comment} you may not use this file except in compliance with the License.
${comment} You may obtain a copy of the License at
${comment}
${comment}     http://www.apache.org/licenses/LICENSE-2.0
${comment}
${comment} Unless required by applicable law or agreed to in writing, software
${comment} distributed under the License is distributed on an \"AS IS\" BASIS,
${comment} WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
${comment} See the License for the specific language governing permissions and
${comment} limitations under the License."
}


# Add a license header into an existing file
add_license_header() {
  # $1: comment character to use in the license
  # $2: fname
  comment=$1
  original=$2
  tmp="./tmpfile"

  # Move original file to tmp
  mv "${original}" "${tmp}"

  # Concat
  get_license_header ${comment} > "${original}"
  echo "" >> "${original}"
  cat "${tmp}" >> "${original}"

  # Clean up
  rm ${tmp}
}


########## DEPENDENCIES ##########
# Download protobuffers library
download_protobuf_lib() {
  assert_folder ${THIRD_PARTY_FOLDER}
  cd ${THIRD_PARTY_FOLDER}

  # Get and unzip library
  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.7.0/protobuf-js-3.7.0.zip
  unzip protobuf-js-3.7.0.zip
  mv protobuf-3.7.0 protobuf

  # Clean
  rm protobuf-js-3.7.0.zip
  cd ..
}


# Downloads proto compiler (binary)
download_protoc() {
  assert_folder ${THIRD_PARTY_FOLDER}
  cd ${THIRD_PARTY_FOLDER}

  assert_folder protoc
  cd protoc

  # Get and unzip binaries
  wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
  unzip protoc-3.6.1-linux-x86_64.zip

  # Get license
  wget https://raw.githubusercontent.com/protocolbuffers/protobuf/master/LICENSE

  # Clean
  rm protoc-3.6.1-linux-x86_64.zip
  cd ../..
}


# Downloads gviz-api.js code
download_gviz_api_js() {
  assert_folder ${THIRD_PARTY_FOLDER}
  cd ${THIRD_PARTY_FOLDER}

  assert_folder gviz
  cd gviz

  # Get code
  wget https://www.google.com/uds/modules/gviz/gviz-api.js

  # Get license
  wget https://raw.githubusercontent.com/GoogleWebComponents/google-chart/master/LICENSE

  # Clean
  cd ../..
}


# Downloads closure library
download_closure_library() {
  assert_folder ${THIRD_PARTY_FOLDER}
  cd ${THIRD_PARTY_FOLDER}

  # Get and unzip library
  wget https://github.com/google/closure-library/archive/v20190301.zip
  unzip v20190301.zip
  mv closure-library-20190301 closure-library

  # Clean
  rm v20190301.zip
  cd ..
}


# Downloads the closure compiler .jar
download_closure_compiler() {
  assert_folder ${THIRD_PARTY_FOLDER}
  cd ${THIRD_PARTY_FOLDER}

  assert_folder closure-compiler
  cd closure-compiler

  # Get and unzip binaries
  wget https://dl.google.com/closure-compiler/compiler-20190121.zip
  unzip compiler-20190121.zip

  # Clean
  rm compiler-20190121.zip
  cd ../..
}


# Downloads all deps
download_all_deps() {
  case "$1" in
    "-c"|"--clean")
      rm -rf "${THIRD_PARTY_FOLDER}"
      ;;
  esac

  download_protobuf_lib
  download_protoc
  download_gviz_api_js
  download_closure_library
  download_closure_compiler
}


# Creates a virtual environment and install py deps
install_py() {
  case "$1" in
    "-c"|"--clean")
      rm -rf ${PY_ENV_FOLDER}
      rm ${PY_ENV_ACTIVATE_FNAME}
      ;;
  esac

  python3 -m virtualenv -p $(which python2) ${PY_ENV_FOLDER}
  source "${PY_ENV_FOLDER}/bin/activate"
  pip install -r requirements.txt

  # Create a link to activate the env later
  ln -s "${PY_ENV_FOLDER}/bin/activate" ${PY_ENV_ACTIVATE_FNAME}
}



########## COMPILE ##########
# Compile css
compile_css() {
  assert_folder ${CSS_BUNDLED_FOLDER}

  # TODO(pdpino): minimize bundled file
  # TODO(pdpino): use multiple .css files as input
  cp eeg_viewer/static/styles/styles.css ${CSS_BUNDLED_FNAME}

  echo "CSS bundled into ${CSS_BUNDLED_FNAME}"
}


# Create a file to use as wrapper on JS compilation,
# to add a license header to the compiled JS
create_temp_js_license_wrapper() {
  get_license_header "//" > "${JS_TEMP_LICENSE_WRAPPER}"
  echo "%output%" >> "${JS_TEMP_LICENSE_WRAPPER}"
}


# Remove license wrapper for cleaning
rm_temp_js_license_wrapper() {
  assert_rm "${JS_TEMP_LICENSE_WRAPPER}"
}


# Compile app_loader.js
compile_js_app_loader() {
  # Remove previously compiled
  assert_rm ${JS_COMPILED_APP_LOADER}

  # Compile
  java -jar ${JS_COMPILER} \
      --js=${JS_APP_LOADER} \
      --output_wrapper_file=${JS_TEMP_LICENSE_WRAPPER} \
      > ${JS_COMPILED_APP_LOADER}

  echo "Compiled JS app_loader into ${JS_COMPILED_APP_LOADER}"
}


# Compile actual app
compile_js_app() {
  # Remove previously compiled
  assert_rm ${JS_COMPILED_APP}

  # Compile
  java -jar ${JS_COMPILER} \
      --js='eeg_viewer/static/js/**.js' \
      --js='!eeg_viewer/static/js/**_test.js' \
      --js='!eeg_viewer/static/js/compiled_*.js' \
      --js='eeg_viewer/jslib/download_helper.js' \
      --js='jsprotos/**.js' \
      --js='third_party/protobuf/js/map.js' \
      --js='third_party/protobuf/js/message.js' \
      --js='third_party/protobuf/js/binary/*.js' \
      --js='!third_party/protobuf/js/binary/*_test.js' \
      --js='third_party/closure-library/closure/goog/**.js' \
      --js='!third_party/closure-library/closure/goog/demos/**.js' \
      --js='!third_party/closure-library/closure/goog/**_test.js' \
      --js='!third_party/closure-library/closure/goog/**_perf.js' \
      --externs='third_party/gviz/gviz-api.js' \
      --entry_point='eeg_viewer/static/js/main.js' \
      --output_wrapper_file=${JS_TEMP_LICENSE_WRAPPER} \
      --force_inject_library es6_runtime \
      -O WHITESPACE_ONLY \
      > ${JS_COMPILED_APP}

  echo "Compiled JS app into ${JS_COMPILED_APP}"
}


# Compile everything JS
compile_js() {
  create_temp_js_license_wrapper

  # Compile app source code
  compile_js_app

  # Compile app_loader, which is directly loaded in HTML.
  # Loads Google Charts JS before loading the app.js
  compile_js_app_loader

  rm_temp_js_license_wrapper
}


# Build JS and CSS code
compile_web() {
  compile_js
  compile_css
}


# Assert that proto compiler exists
assert_protoc() {
  if [[ ! -f "${PROTO_COMPILER_CMD}" ]]; then
    echo "ERROR: Proto compiler not found in ${PROTO_COMPILER_CMD}, not compiling"
    echo "Try downloading dependencies first"
    exit 1
  fi
}


# Compile JS protos
compile_js_protos() {
  # Assert folder
  assert_folder "${PROTO_JS_COMPILED_FOLDER}"

  # Compile
  ${PROTO_COMPILER_CMD} -I="${PROTO_SRC_FOLDER}" --js_out=library="${PROTO_JS_COMPILED_FOLDER}/main",binary:. protos/*.proto
  ${PROTO_COMPILER_CMD} --js_out=library="${PROTO_JS_COMPILED_FOLDER}/google",binary:. ${PROTO_TIMESTAMP_FNAME}

  # Add license header
  add_license_header ${JS_COMMENT} "${PROTO_JS_COMPILED_FOLDER}/main.js"
  add_license_header ${JS_COMMENT} "${PROTO_JS_COMPILED_FOLDER}/google.js"

  echo "JS Protos compiled into ${PROTO_JS_COMPILED_FOLDER}"
}


# Compile Python protos
compile_py_protos() {
  # Assert folder
  assert_folder "${PROTO_PY_COMPILED_FOLDER}"

  # Create empty __init__ file
  echo "" > "${PROTO_PY_COMPILED_FOLDER}/__init__.py"

  # Compile
  ${PROTO_COMPILER_CMD} -I="${PROTO_SRC_FOLDER}" --python_out="${PROTO_PY_COMPILED_FOLDER}" protos/*.proto

  # Add license header
  for name in $(ls "${PROTO_PY_COMPILED_FOLDER}"); do
    compiled_proto="${PROTO_PY_COMPILED_FOLDER}/${name}"
    [[ -f "${compiled_proto}" ]] && add_license_header "${PY_COMMENT}" "${compiled_proto}"
  done

  echo "Python Protos compiled into ${PROTO_PY_COMPILED_FOLDER}"
}


# Compile proto into JS and python
compile_protos() {
  assert_protoc

  compile_js_protos
  compile_py_protos
}


########## TESTS ##########
# Run all the python tests
run_py_tests() {
  # Move to google_research/ folder
  cd ..

  echo "############## PYTHON TESTS ###############"
  for test_file in $(ls eeg_modelling/eeg_viewer/*_test.py); do
    test_name=$(basename ${test_file})
    test_name="${test_name%.*}" # Remove extension
    python -m eeg_modelling.eeg_viewer."${test_name}"
  done
  echo "############## END OF PYTHON TESTS ###############"
}


# Run all JS tests
run_js_tests() {
  echo "JS tests not implemented yet"
}


# Run both python and JS tests
run_all_tests() {
  run_py_tests
  run_js_tests
}




########## MAIN ##########
# Run EEG viewer server
run_server() {
  case "$1" in
    "-r"|"--recompile")
      compile_web
      shift
      ;;
  esac

  # Move to google_research/
  cd ..

  # Run the server
  python -m eeg_modelling.eeg_viewer.server "$*"
}


# Ensure that the working directory is eeg_modelling/
assert_working_dir() {
  wd=`basename $(pwd)`
  case $wd in
    "google_research"|"google-research")
      cd eeg_modelling
      ;;
    "eeg_modelling")
      ;;
    *)
      echo "ERROR: This script should be called from google-research/ or eeg_modelling/ folders"
      exit 1
  esac
}


assert_working_dir

cmd=$1
[[ $# -gt 0 ]] && shift # shift to next argument, if any

case "$cmd" in
  run)
    run_server "$*"
    ;;
  build)
    compile_web
    ;;
  js|compile_js)
    compile_js
    ;;
  css|compile_css)
    compile_css
    ;;
  test|tests)
    run_all_tests
    ;;
  proto|protos|compile_protos)
    compile_protos
    ;;
  download)
    download_all_deps "$*"
    ;;
  install_py)
    install_py "$*"
    ;;
  *)
    echo "Usage:   $0 [cmd] [cmd_options]"
    echo "Commands:"
    echo "      run                       - run server"
    echo "      build                     - compiles JS and CSS"
    echo "      js, compile_js            - compile javascript code"
    echo "      css, compile_css          - bundle css"
    echo "      test                      - run tests"
    echo "      protos, compile_protos    - compile proto into JS and python"
    echo "      download                  - download all dependencies"
    echo "      install_py                - create a virtual env and"
    echo "                                  install py dependencies"
    echo "      help, --help              - this help message"
    ;;
esac
