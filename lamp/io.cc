// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "io.h"  // NOLINT(build/include)

#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <sys/types.h>

#include <cstring>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

void ToTrails(const SplitStringTrails& string_trails, int min_user_count,
              bool split_by_time, Trails* train_trails, Trails* test_trails,
              int* num_items) {
  std::unordered_map<std::string, int> frequency;
  for (const SplitStringTrail& string_trail : string_trails) {
    std::unordered_set<std::string> user_items_seen;
    for (const std::string& item : string_trail.trail) {
      // Count unique users only if requested
      if (split_by_time || user_items_seen.count(item) == 0) {
        frequency[item]++;
      }
      user_items_seen.insert(item);
    }
  }
  train_trails->clear();
  test_trails->clear();
  int num_frequent_locations = 0;
  std::unordered_map<std::string, int> location_index;
  int num_check_ins_at_frequent_locations = 0,
      num_check_ins_at_infrequent_locations = 0;
  for (const SplitStringTrail& string_trail : string_trails) {
    Trail trail;
    for (const std::string& location_id : string_trail.trail) {
      int loc_index = 0;  // Infrequent location are mapped to 0.
      if (frequency.at(location_id) >= min_user_count) {
        loc_index = location_index[location_id];
        // This is the first time we are seeing this place.
        if (loc_index == 0) {
          loc_index = ++num_frequent_locations;
          location_index[location_id] = loc_index;
        }
        num_check_ins_at_frequent_locations++;
      } else {
        num_check_ins_at_infrequent_locations++;
      }
      // TODO(stamas): Maybe remove self-loops?
      trail.push_back(loc_index);
    }
    if (trail.empty()) {
      continue;
    }
    int trail_size = trail.size();
    int train_size = split_by_time ? string_trail.train_size : trail_size;
    die_if(train_size > trail_size, "Train size too large in ToTrails");
    if (train_size > 0) {
      train_trails->emplace_back(trail.begin(), trail.begin() + train_size);
    }
    if (train_size != trail_size) {
      test_trails->emplace_back(trail.begin() + train_size, trail.end());
    }
  }
  *num_items = num_frequent_locations + 1;
  printf(
      "Saw %d frequent items with %d total visits and %d visits of "
      "infrequent items\n",
      num_frequent_locations, num_check_ins_at_frequent_locations,
      num_check_ins_at_infrequent_locations);
}

void ChopNewLine(char* line) {
  int len = std::strlen(line);
  if (len > 0) {
    line[len - 1] = '\0';
  }
}

bool ParseBrightkiteLine(const char* line, int line_no, int* user_id,
                         std::string* time, std::string* location_id) {
  static int num_warnings = 0;
  char time_buffer[256], lat_buffer[256], lng_buffer[256],
      location_id_buffer[256];
  int num_parsed = sscanf(line, "%d\t%s\t%s\t%s\t%s", user_id, time_buffer,
                          lat_buffer, lng_buffer, location_id_buffer);
  if (num_parsed != 5 && ++num_warnings <= -1) {
    num_warnings++;
    printf("Skipped line number %d: expected 5 parsed %d items in: %s", line_no,
           num_parsed, line);
  }
  *time = time_buffer;
  *location_id = location_id_buffer;
  return num_parsed == 5;
}

SplitStringTrails ReadBrightkite(const std::string& file_name,
                                 const std::string& max_train_time) {
  FILE* file = fopen(file_name.c_str(), "r");
  die_if(!file, "Can't open", file_name);
  StringTrail trail;
  SplitStringTrails trails;
  int train_size = 0;
  int prev_user_id = -1;
  std::string prev_time;
  int num_check_ins = 0;
  int num_parse_skipped = 0, num_zero_location_skipped = 0,
      num_self_loops_skipped = 0, num_time_skipped = 0;
  char line[1024];
  while (fgets(line, sizeof(line), file)) {
    std::string time;
    int user_id;
    std::string location_id;
    if (!ParseBrightkiteLine(line, ++num_check_ins, &user_id, &time,
                             &location_id)) {
      ++num_parse_skipped;
      continue;
    }
    if (user_id != prev_user_id && !trail.empty()) {
      // Check-ins are in reverse chronological order
      trails.push_back({StringTrail(trail.rbegin(), trail.rend()),
                        max_train_time.empty() ? -1 : train_size});
      trail.clear();
      train_size = 0;
    }
    prev_user_id = user_id;
    if (location_id == "00000000000000000000000000000000") {
      num_zero_location_skipped++;
      continue;
    }
    if (!trail.empty() && trail.back() == location_id) {
      num_self_loops_skipped++;
      continue;
    }
    // Lex cmp works for timestamp format
    if (!trail.empty() && prev_time < time) {
      if (++num_time_skipped < -1) {
        printf(
            "Skipped Brightkite line due to increasing time %s time %s "
            "prev_time %s\n",
            line, time.c_str(), prev_time.c_str());
      }
      continue;
    }
    if (time <= max_train_time) {
      train_size++;
    }
    prev_time = time;
    trail.push_back(location_id);
  }
  if (!trail.empty()) {
    trails.push_back({StringTrail(trail.rbegin(), trail.rend()),
                      max_train_time.empty() ? -1 : train_size});
  }
  die_if(ferror(file), "Error while reading", file_name);
  printf("Read %d Brightkite check-ins in %zu trails\n", num_check_ins,
         trails.size());
  printf(
      "Skipped %d lines with parse errors, %d with 000* location, %d with "
      "self-loops, %d with increasing time\n",
      num_parse_skipped, num_zero_location_skipped, num_self_loops_skipped,
      num_time_skipped);
  die_if(fclose(file) != 0, "Can't close", file_name);
  return trails;
}

std::vector<std::string> Tokenize(const std::string& s, char delimiter) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delimiter)) {
    elems.push_back(item);
  }
  return elems;
}

StringTrail ParseWikiLine(const std::string& line, int* num_back_buttons) {
  StringTrail trail, page_stack;
  for (const std::string& elem : Tokenize(line, ';')) {
    std::string item;
    if (elem == "<") {  // back button
      die_if(page_stack.empty(),
             std::string("Too many back steps in Wiki line: ") + line);
      item = page_stack.back();
      page_stack.pop_back();
      *num_back_buttons += 1;
    } else {
      item = elem;
      page_stack.push_back(elem);
    }
    trail.push_back(item);
  }
  return trail;
}

SplitStringTrails ReadWiki(const std::string& file_name) {
  FILE* file = fopen(file_name.c_str(), "r");
  die_if(!file, "Can't open", file_name);
  SplitStringTrails trails;
  char line[2048];
  int sum_length = 0, num_back_buttons = 0;
  while (fgets(line, sizeof(line), file)) {
    ChopNewLine(line);
    StringTrail trail = ParseWikiLine(line, &num_back_buttons);
    sum_length += trail.size();
    trails.push_back({trail, -1});
  }
  die_if(ferror(file), "Error while reading", file_name);
  printf("Read %zu Wiki trails with %d total visits and %d back steps\n",
         trails.size(), sum_length, num_back_buttons);
  die_if(fclose(file) != 0, "Can't close", file_name);
  return trails;
}

void ReadReutersFilesInto(bool is_training, SplitStringTrails* trails) {
  std::string dir_name =
      is_training ? "data/reuters/training" : "data/reuters/test";
  DIR* dirp = opendir(dir_name.c_str());
  struct dirent* dp;
  dir_name += '/';
  int num_files = 0;
  int num_words = 0;
  while ((dp = readdir(dirp)) != NULL) {
    if (strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0) {
      continue;
    }
    ++num_files;
    // printf("%s\n", dp->d_name);
    SplitStringTrails document_words = ReadText(dir_name + dp->d_name, false);
    num_words += document_words.front().trail.size();
    int train_size = is_training ? document_words.front().trail.size() : 0;
    trails->push_back({document_words.front().trail, train_size});
  }
  closedir(dirp);
  printf("Read %d Reuters %s files with %d words in total\n", num_files,
         is_training ? "training" : "testing", num_words);
}

SplitStringTrails ReadReuters() {
  SplitStringTrails ret;
  ReadReutersFilesInto(true, &ret);
  ReadReutersFilesInto(false, &ret);
  return ret;
}

SplitStringTrails ReadText(const std::string& file_name, bool verbose) {
  FILE* file = fopen(file_name.c_str(), "r");
  die_if(!file, "Can't open", file_name);
  StringTrail trail;
  std::string word;
  int c;
  while ((c = fgetc(file)) != EOF) {
    if (isalpha(c)) {
      word += tolower(c);
    } else {
      if (word.size() >= 2) {
        trail.push_back(word);
      }
      word.clear();
    }
  }
  die_if(ferror(file), "Error while reading", file_name);
  if (verbose) {
    printf("Read text with %zu words\n", trail.size());
  }
  die_if(fclose(file) != 0, "Can't close", file_name);
  // Use the same data for testing
  // trail.insert(trail.end(), trail.begin(), trail.end());
  // Single trail split into train and test halves.
  return {{trail, static_cast<int>(trail.size() / 2)}};
}

bool ParseLastfmLine(const char* line, int line_no, int* user_id,
                     std::string* time, std::string* artist) {
  static int num_warnings = 0;
  const int kMaxWarnings = 10;
  std::vector<std::string> fields = Tokenize(line, '\t');

  if (fields.size() != 6 && ++num_warnings <= kMaxWarnings) {
    printf("Skipped line number %d: expected 6 fields, found %zu in: %s",
           line_no, fields.size(), line);
    return false;
  }
  if (sscanf(fields[0].c_str(), "user_%d", user_id) != 1) {
    if (++num_warnings <= kMaxWarnings) {
      printf("Skipped line number %d: can't parse user id in 1st field in: %s",
             line_no, line);
    }
    return false;
  }
  *time = fields[1];
  *artist = fields[3];  // artist name, artist id is sometimes missing
  return true;
}

SplitStringTrails ReadLastfm(const std::string& file_name, int max_users,
                             const std::string& max_train_time) {
  FILE* file = fopen(file_name.c_str(), "r");
  die_if(!file, "Can't open", file_name);
  StringTrail trail;
  SplitStringTrails trails;
  int train_size = 0;
  int prev_user_id = -1;
  std::string prev_time;
  int num_plays = 0;
  int num_parse_skipped = 0, num_empty_artist_skipped = 0,
      num_self_loops_skipped = 0, num_time_skipped = 0;
  char line[1024];
  while (fgets(line, sizeof(line), file)) {
    std::string time;
    int user_id;
    std::string artist;
    if (!ParseLastfmLine(line, ++num_plays, &user_id, &time, &artist)) {
      ++num_parse_skipped;
      continue;
    }
    if (user_id != prev_user_id && !trail.empty()) {
      // Plays are in reverse chronological order
      trails.push_back({StringTrail(trail.rbegin(), trail.rend()),
                        max_train_time.empty() ? -1 : train_size});
      trail.clear();
      train_size = 0;
      if (static_cast<int>(trails.size()) >= max_users) {
        break;
      }
    }
    prev_user_id = user_id;
    if (artist.empty()) {
      num_empty_artist_skipped++;
      continue;
    }
    if (!trail.empty() && trail.back() == artist) {
      num_self_loops_skipped++;
      continue;
    }
    // Lex cmp works for timestamp format
    if (!trail.empty() && prev_time < time) {
      if (++num_time_skipped < -1) {
        printf(
            "Skipped Last.fm line due to increasing time %s time %s prev_time "
            "%s\n",
            line, time.c_str(), prev_time.c_str());
      }
      continue;
    }
    if (time <= max_train_time) {
      train_size++;
    }
    prev_time = time;
    trail.push_back(artist);
  }
  if (!trail.empty()) {
    trails.push_back({StringTrail(trail.rbegin(), trail.rend()),
                      max_train_time.empty() ? -1 : train_size});
  }
  die_if(ferror(file), "Error while reading", file_name);
  printf("Read %d Lastfm plays in %zu trails\n", num_plays, trails.size());
  printf(
      "Skipped %d lines with parse errors, %d with empty artist, %d with "
      "self-loops, %d with increasing time\n",
      num_parse_skipped, num_empty_artist_skipped, num_self_loops_skipped,
      num_time_skipped);
  die_if(fclose(file) != 0, "Can't close", file_name);
  return trails;
}
