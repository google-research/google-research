# Copyright 2020 The Google Research Authors.
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
#
# Run the wikinews parser on fake data as a regression test.
set -e
set -x

TEMP_DIR=$(mktemp -d)

# Create a fake mentions.tsv for testing.
cat >"${TEMP_DIR}/mentions.tsv" <<EOF
docid	position	length	mention	url
2018-04-10_00_Google	36	6	Google	https://en.wikipedia.org/wiki/Google
2018-04-10_00_Google	67	3	NER	https://en.wikipedia.org/wiki/Named-entity_recognition
EOF

# Create a fake docs.tsv for testing.
cat >"${TEMP_DIR}/docs.tsv" <<EOF
docid	date	url	title	wiki_md5	text_md5
2018-04-10_00_Google	2018-04-10	https://en.wikinews.org/wiki/Google_researchers_publish	Google researchers publish	f28135b386a9f467ac3981dc2fa6b426	07394ee0f982f33025bfbd82aff6a0e3
EOF

# Create a fake wikinews archive for testing.
cat >"${TEMP_DIR}/pages.xml" <<EOF
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.mediawiki.org/xml/export-0.10/ http://www.mediawiki.org/xml/export-0.10.xsd" version="0.10" xml:lang="en">
  <page>
    <title>Google researchers publish</title>
    <revision>
      <text xml:space="preserve">{{date|April 10, 2018}}
On Tuesday {{w|Google}} researchers published a [[Named-entity_recognition|NER]] paper.

{{haveyoursay}}

== Sources == 

{{Publish}}
{{Archive}}</text>
    </revision>
  </page>
</mediawiki>
EOF

# Bzip the fake archive.
bzip2 "${TEMP_DIR}/pages.xml"

virtualenv -p python3 .
source ./bin/activate

# Run the parser on the fake input created above.
pip install -r dense_representations_for_entity_retrieval/requirements.txt
python -m dense_representations_for_entity_retrieval.parse_wikinews \
   --wikinews_archive="${TEMP_DIR}/pages.xml.bz2" \
   --tsv_dir="${TEMP_DIR}" \
   --output_dir="${TEMP_DIR}" \
   --logtostderr
