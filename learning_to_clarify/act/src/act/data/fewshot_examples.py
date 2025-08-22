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

pacific_intent_summarization_examples = [
    """The following is a conversation between a User and an Assistant. The User is asking some questions. Summarize what information the User is looking for.
[context]
During the period 5.0 million (2017/18: 5.0 million) options were granted under the Sharesave Plan, with a weighted average exercise price at the date of exercise of 30 pence per ordinary share (2017/18: 33 pence). The options outstanding at 30 March 2019 had a weighted average exercise price of 32 pence (2017/18: 33 pence), and a weighted average remaining contractual life of 1.6 years (2017/18: 1.6 years). In 2018/19, the Group recognised an expense of £2.1m (2017/18: £2.8m), related to all equity-settled share-based payment transactions. Premier Foods plc Sharesave Plan
[data]
 : 2018/19 | | 2017/18 |
 : | Weighted average exercise price | | Weighted average exercise price
 : Options | (p) | Options | (p)
Outstanding at the beginning of the period : 17,835,628 | 33 | 20,231,334 | 35
Exercised during the period : (4,306,470) | 32 | (3,536,539) | 34
Granted during the period : 5,022,240 | 30 | 4,988,669 | 33
Forfeited/lapsed during the period : (2,447,511) | 33 | (3,847,836) | 44
Outstanding at the end of the period : 16,103,887 | 32 | 17,835,628 | 33
Exercisable at the end of the period : 2,673,154 | 32 | 792,451 | 35
User: What was the amount of options granted under the Sharesave plan?
Assistant: 5.0 million
User: What was the weighted average exercise price of outstanding options at 30 March 2019?
Assistant: 32 pence
User: What was the expense recognised in 2018/19?
Assistant: £2.1m
User: What was the change in the Outstanding for options from 2017/18 to 2018/19?
Assistant: What kind of outstanding are you asking about?
User: Outstanding at the beginning of the period.
Assistant: -2395706
[Information] The user wants to know: 1. what was the amount of options granted under the Sharesave plan, 2. what was the weighted average exercise price of outstanding options at 30 March 2019, 3. What was the expense recognised in 2018/19, 4. What was the change in the Outstanding for options  at the beginning of the period from 2017/18 to 2018/19.""",
    """The following is a conversation between a User and an Assistant. The User is asking some questions. Summarize what information the User is looking for.
[context]
Unaudited Pro Forma Financial Information The pro forma financial information in the table below presents the combined results of operations for ACI and Speedpay as if the acquisition had occurred January 1, 2018. The pro forma information is shown for illustrative purposes only and is not necessarily indicative of future results of operations of the Company or results of operations of the Company that would have actually occurred had the transaction been in effect for the periods presented. This pro forma information is not intended to represent or be indicative of actual results had the acquisition occurred as of the beginning of each period, and does not reflect potential synergies, integration costs, or other such costs or savings. Certain pro forma adjustments have been made to net income (loss) for the year ended December 31, 2019 and 2018, to give effect to estimated adjustments that remove the amortization expense on eliminated Speedpay historical identifiable intangible assets, add amortization expense for the value of acquired identified intangible assets (primarily acquired software, customer relationships, and trademarks), and add estimated interest expense on the Company’s additional Delayed Draw Term Loan and Revolving Credit Facility borrowings. Additionally, certain transaction expenses that are a direct result of the acquisition have been excluded from the year ended December 31, 2019. The following is the unaudited summarized pro forma financial information for the periods presented (in thousands, except per share data): Walletron On May 9, 2019, the Company also completed the acquisition of Walletron, which delivers patented mobile wallet technology. The Company has included the financial results of Walletron in the consolidated financial statements from the date of acquisition, which were not material. RevChip and TranSend On October 1, 2019, the Company acquired certain technology assets of RevChip, LLC ("RevChip") and TranSend Integrated Technologies Inc. ("TranSend") for a combined $7.0 million. As substantially all of the value was in the developed technology, the purchase was recognized as an asset acquisition. The Company has included the financial results of RevChip and TranSend in the consolidated financial statements from the date of acquisition, which were not material.
[data]
 : Years Ended December 31, |
 : 2019 | 2018
Pro forma revenue : $1,382,957 | $1,361,729
Pro forma net income : $82,003 | $88,428
Pro forma income per share: : |
Basic : $ 0.71 | $ 0.76
Diluted : $ 0.69 | $ 0.75
[conversation]
User: What does Walletron deliver?
Assistant: patented mobile wallet technology.
User: How much did it cost the company to acquire certain technology assets of RevChip, LLC ("RevChip") and TranSend Integrated Technologies Inc. ("TranSend")?
Assistant: $7.0 million
User: What was the pro forma revenue in 2019?
Assistant: $1,382,957
User: What was the change in its amount between 2018 and 2019?
Assistant: 21228
User: How about that in pro forma net income?
Assistant: -6425
[Information] The user wants to know: 1. What technology Walletron delivers, 2. How much it cost Walletron to acquire certain technology assets of RevChip, LLC and TranSend Integrated Technologies Inc., 3. What the pro forma revenue was in 2019, 4. What the change in pro forma revenue was between 2018 and 2019, and 5. What the change in pro forma net income was between 2018 and 2019.""",
    """The following is a conversation between a User and an Assistant. The User is asking some questions. Summarize what information the User is looking for.
[context]
Lines of Credit The following table summarizes our available lines of credit and committed and uncommitted lines of credit, including the revolving credit facility discussed above, and the amounts available under our accounts receivable securitization programs. (1) Includes total borrowings under the accounts receivable securitization programs, the revolving credit facility and borrowings under lines of credit available to several subsidiaries. (2) Of the total available lines of credit, $1,137.4 million were committed as of December 31, 2019.
[data]
 : December 31, |
(In millions) : 2019 | 2018
Used lines of credit (1) : $ 98.9 | $ 232.8
Unused lines of credit : 1,245.2 | 1,135.3
Total available lines of credit(2) : $ 1,344.1 | $ 1,368.1
[conversation]
User: What does the table show?
Assistant: summarizes our available lines of credit and committed and uncommitted lines of credit, including the revolving credit facility discussed above, and the amounts available under our accounts receivable securitization programs.
User: How much was commited as of December 31, 2019 of total available lines of credit?
Assistant: $1,137.4 million
User: How much money has not been committed as of that?
Assistant: 206.7
User: What is the percentage of used lines of credit to that?
Assistant: 7.36
User: What is the difference between the credit for 2019 and 2018?
Assistant: What kind of credits are you asking for?
User: Unused lines of credit.
[Information] The user wants to know: 1. what information the table conveys, 2. how much money was committed as of December 31, 2019 in terms of total available lines of credit, 3. how much money has not been committed as of December 31, 2019,  4.what is the percentage of used lines of credit relative to the amount of uncommitted money, and 5. the difference between unused lines of credit between 2019 and 2018.""",
]

pacific_user_simulator_examples = [
    """The following is a conversation between a User and an Assistant. The User is asking some questions. The user wants to know: 1. what was the amount of options granted under the Sharesave plan, 2. what was the weighted average exercise price of outstanding options at 30 March 2019, 3. What was the expense recognised in 2018/19, 4. What was the change in the Outstanding for options  at the beginning of the period from 2017/18 to 2018/19.
[context]
During the period 5.0 million (2017/18: 5.0 million) options were granted under the Sharesave Plan, with a weighted average exercise price at the date of exercise of 30 pence per ordinary share (2017/18: 33 pence). The options outstanding at 30 March 2019 had a weighted average exercise price of 32 pence (2017/18: 33 pence), and a weighted average remaining contractual life of 1.6 years (2017/18: 1.6 years). In 2018/19, the Group recognised an expense of £2.1m (2017/18: £2.8m), related to all equity-settled share-based payment transactions. Premier Foods plc Sharesave Plan
[data]
 : 2018/19 | | 2017/18 |
 : | Weighted average exercise price | | Weighted average exercise price
 : Options | (p) | Options | (p)
Outstanding at the beginning of the period : 17,835,628 | 33 | 20,231,334 | 35
Exercised during the period : (4,306,470) | 32 | (3,536,539) | 34
Granted during the period : 5,022,240 | 30 | 4,988,669 | 33
Forfeited/lapsed during the period : (2,447,511) | 33 | (3,847,836) | 44
Outstanding at the end of the period : 16,103,887 | 32 | 17,835,628 | 33
Exercisable at the end of the period : 2,673,154 | 32 | 792,451 | 35
[conversation]
User: What was the amount of options granted under the Sharesave plan?
Assistant: 5.0 million
User: What was the weighted average exercise price of outstanding options at 30 March 2019?
Assistant: 32 pence
User: What was the expense recognised in 2018/19?
Assistant: £2.1m
User: What was the change in the Outstanding for options from 2017/18 to 2018/19?
Assistant: What kind of outstanding are you asking about?
User: Outstanding at the beginning of the period.
Assistant: -2395706""",
    """The following is a conversation between a User and an Assistant. The User is asking some questions. The user wants to know: 1. What technology Walletron delivers, 2. How much it cost Walletron to acquire certain technology assets of RevChip, LLC and TranSend Integrated Technologies Inc., 3. What the pro forma revenue was in 2019, 4. What the change in pro forma revenue was between 2018 and 2019, and 5. What the change in pro forma net income was between 2018 and 2019.
[context]
Unaudited Pro Forma Financial Information The pro forma financial information in the table below presents the combined results of operations for ACI and Speedpay as if the acquisition had occurred January 1, 2018. The pro forma information is shown for illustrative purposes only and is not necessarily indicative of future results of operations of the Company or results of operations of the Company that would have actually occurred had the transaction been in effect for the periods presented. This pro forma information is not intended to represent or be indicative of actual results had the acquisition occurred as of the beginning of each period, and does not reflect potential synergies, integration costs, or other such costs or savings. Certain pro forma adjustments have been made to net income (loss) for the year ended December 31, 2019 and 2018, to give effect to estimated adjustments that remove the amortization expense on eliminated Speedpay historical identifiable intangible assets, add amortization expense for the value of acquired identified intangible assets (primarily acquired software, customer relationships, and trademarks), and add estimated interest expense on the Company’s additional Delayed Draw Term Loan and Revolving Credit Facility borrowings. Additionally, certain transaction expenses that are a direct result of the acquisition have been excluded from the year ended December 31, 2019. The following is the unaudited summarized pro forma financial information for the periods presented (in thousands, except per share data): Walletron On May 9, 2019, the Company also completed the acquisition of Walletron, which delivers patented mobile wallet technology. The Company has included the financial results of Walletron in the consolidated financial statements from the date of acquisition, which were not material. RevChip and TranSend On October 1, 2019, the Company acquired certain technology assets of RevChip, LLC ("RevChip") and TranSend Integrated Technologies Inc. ("TranSend") for a combined $7.0 million. As substantially all of the value was in the developed technology, the purchase was recognized as an asset acquisition. The Company has included the financial results of RevChip and TranSend in the consolidated financial statements from the date of acquisition, which were not material.
[data]
 : Years Ended December 31, |
 : 2019 | 2018
Pro forma revenue : $1,382,957 | $1,361,729
Pro forma net income : $82,003 | $88,428
Pro forma income per share: : |
Basic : $ 0.71 | $ 0.76
Diluted : $ 0.69 | $ 0.75
[conversation]
User: What does Walletron deliver?
Assistant: patented mobile wallet technology.
User: How much did it cost the company to acquire certain technology assets of RevChip, LLC ("RevChip") and TranSend Integrated Technologies Inc. ("TranSend")?
Assistant: $7.0 million
User: What was the pro forma revenue in 2019?
Assistant: $1,382,957
User: What was the change in its amount between 2018 and 2019?
Assistant: 21228
User: How about that in pro forma net income?
Assistant: -6425""",
    """The following is a conversation between a User and an Assistant. The User is asking some questions. The user wants to know: 1. what information the table conveys, 2. how much money was committed as of December 31, 2019 in terms of total available lines of credit, 3. how much money has not been committed as of December 31, 2019,  4.what is the percentage of used lines of credit relative to the amount of uncommitted money, and 5. the difference between unused lines of credit between 2019 and 2018.
[context]
Lines of Credit The following table summarizes our available lines of credit and committed and uncommitted lines of credit, including the revolving credit facility discussed above, and the amounts available under our accounts receivable securitization programs. (1) Includes total borrowings under the accounts receivable securitization programs, the revolving credit facility and borrowings under lines of credit available to several subsidiaries. (2) Of the total available lines of credit, $1,137.4 million were committed as of December 31, 2019.
[data]
 : December 31, |
(In millions) : 2019 | 2018
Used lines of credit (1) : $ 98.9 | $ 232.8
Unused lines of credit : 1,245.2 | 1,135.3
Total available lines of credit(2) : $ 1,344.1 | $ 1,368.1
[conversation]
User: What does the table show?
Assistant: summarizes our available lines of credit and committed and uncommitted lines of credit, including the revolving credit facility discussed above, and the amounts available under our accounts receivable securitization programs.
User: How much was commited as of December 31, 2019 of total available lines of credit?
Assistant: $1,137.4 million
User: How much money has not been committed as of that?
Assistant: 206.7
User: What is the percentage of used lines of credit to that?
Assistant: 7.36
User: What is the difference between the credit for 2019 and 2018?
Assistant: What kind of credits are you asking for?
User: Unused lines of credit.""",
]
