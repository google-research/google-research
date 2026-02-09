# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

SUCCESSFUL_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```
"""

FAILED_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```
"""

PARALLEL_SI = """
You are an expert in web navigation. You will be given a user query and multiple trajectories showing how an agent attempted the task.
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to **compare and contrast** these trajectories to identify the most useful and generalizable strategies as memory items.
Use **self-contrast reasoning**:
  - Identify patterns and strategies that consistently led to success.
  - Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
  - Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
  - Think first: Why did some trajectories succeed while others failed?
  - You can extract *at most 5* memory items from all trajectories combined.
  - Do not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
  - Make sure each memory item captures **actionable** and **transferable** insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
```
"""

PARALLEL_AWM_SI = """
You are an expert in web navigation. You will be given a user query and multiple trajectories showing how an agent attempted the task.
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to extract the common workflows to solve these tasks.
**Workflow extraction process**:
  - Examine all given trajectories (both successful and failed).
  - Look for repetitive subsets of actions that occur across multiple tasks.
  - Focus on action sequences that are critical to completing parts of the task, not trivial or one-off actions.

## Important notes
  - Think first: Why did some trajectories succeed while others failed?
  - You can extract *at most 5* workflow items from all trajectories combined.
  - Do not repeat similar or overlapping items.
  - **Each workflow must have at least 2 steps (i.e., 2 <think><action> sequences)**.
  - Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
  - Ensure workflows are actionable and can be reused in similar tasks.

## Output Format
Your output must strictly follow the following example format:

```
Workflow i: Find all orders
<think>
To find the date of the first purchase, I need to navigate to the order history section of the user's account. I will start by clicking on the "My Account" link.
</think>
<action>
click('227')
</action>

<think>
To find the date of the first purchase, I need to navigate to the "My Orders" section where the order history is listed. From there, I can look for the earliest order date. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('1843')
</action>
```
"""

SEQUENTIAL_PROMPT = """
Important: Let's carefully re-examine the previous trajectory, including your reasoning steps and actions taken.
Pay special attention to whether you used the correct elements on the page, and whether your response addresses the user query.
If you find inconsistencies, correct them. If everything seems correct, confirm your final answer.
Output must stay in the same “<think>...</think><action></action>” format as previous trajectories.
"""

SEQUENTIAL_FOLLOWING_PROMPT = """
Let's check again.
Output must stay in the same “<think>...</think><action></action>” format as previous trajectories.
"""

AWM_INSTRUCTION = """
Given a list of web navigation tasks, your task is to extract the common workflows to solve these tasks.
Each given task contains a natural language instruction, and a series of actions to solve the task. You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping workflows. Each workflow should have at least two steps. Represent the non-fixed elements (input text, button strings) with descriptive variable names as shown in the example.
Keep the values of invariant elements, e.g., id of "Search" or "Customers", as they will share and stay invariant across tasks.
Try to generate as many workflows that can cover all the tasks in the input list.
"""

AWM_EXAMPLE = """
## Concrete Examples

Query: What is the date when I made my first purchase on this site?
Actions:
<think>
To find the date of the first purchase, I need to navigate to the order history section of the user's account. I will start by clicking on the "My Account" link.
</think>
<action>
click('227')
</action>

<think>
To find the date of the first purchase, I need to navigate to the "My Orders" section where the order history is listed. From there, I can look for the earliest order date. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('1843')
</action>

<think>
To find the date of the first purchase, I need to look at the earliest date in the "Date" column of the "Orders" table. The current page shows the first 10 orders, but there are multiple pages. I will navigate to the last page to find the earliest order date.
</think>
<action>
click('1788')
</action>

<think>
To find the date of the first purchase, I need to look for the earliest date in the "Date" column of the "My Orders" table. From the current view, the earliest date visible is "5/2/23". However, there might be more orders on other pages. I will navigate to the last page to ensure I find the earliest date.
</think>
<action>
click('1784')
</action>

<think>
To find the earliest purchase date, I need to look at the "Date" column in the "Orders" table and identify the earliest date listed. The dates currently visible are: 4/27/22, 4/5/22, 3/10/22, 3/2/22, 3/2/22. The earliest date visible is 3/2/22. I will respond with this date.
</think>
<action>
send_msg_to_user('The earliest purchase date is 3/2/22.')
</action>

Query: I want to browse the products in the Cabinets, Racks & Shelves category
Actions:
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Home & Kitchen" section, which likely contains the Cabinets, Racks & Shelves category. I will start by hovering over the "Home & Kitchen" menu item to reveal the subcategories.
</think>
<action>
hover('735')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Storage & Organization" section under the "Home & Kitchen" menu. I will start by hovering over the "Storage & Organization" menu item to reveal the subcategories.
</think>
<action>
hover('786')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to click on the "Racks, Shelves & Drawers" link under the "Storage & Organization" menu. This will take me to the desired category.
</think>
<action>
click('797')
</action>

<think>
The current page is displaying the "Racks, Shelves & Drawers" category under "Storage & Organization" in the "Home & Kitchen" section. The user wants to browse products in the "Cabinets, Racks & Shelves" category. Since "Racks, Shelves & Drawers" is already a subcategory of "Storage & Organization," I will confirm with the user if this is the correct category they want to browse.
</think>
<action>
send_msg_to_user('You are currently viewing the "Racks, Shelves & Drawers" category under "Storage & Organization." Is this the correct category you want to browse, or are you looking for a different category?')
</action>


Query: Get the order number of my most recent cancelled order
Actions:
<think>
To get the order number of the most recent cancelled order, I need to navigate to the "My Account" section where order history is typically managed. From the current page, I can see the "My Account" link in the top right corner. I will click on the "My Account" link to proceed.
</think>
<action>
click('227')
</action>

<think>
To get the order number of the most recent cancelled order, I need to navigate to the "My Orders" section. From there, I can filter or sort the orders to find the most recent cancelled order. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('1843')
</action>

<think>
To get the order number of the most recent cancelled order, I need to look for the order with the status "Canceled" and note its order number. From the current page, I can see that the most recent cancelled order is order number 000000170.
</think>
<action>
send_msg_to_user('The order number of your most recent cancelled order is 000000170.')
</action>


## Summary Workflows

Workflow 1: Find all orders
<think>
To find the date of the first purchase, I need to navigate to the order history section of the user's account. I will start by clicking on the "My Account" link.
</think>
<action>
click('my_account_id')
</action>

<think>
To find the date of the first purchase, I need to navigate to the "My Orders" section where the order history is listed. From there, I can look for the earliest order date. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('my_orders_id')
</action>

Workflow 2: Obtain order information, such as date or order number
<think>
To find the earliest purchase date, I need to look at the "Date" column in the "Orders" table and identify the earliest date listed. The dates currently visible are: 4/27/22, 4/5/22, 3/10/22, 3/2/22, 3/2/22. The earliest date visible is 3/2/22. I will respond with this date.
</think>
<action>
send_msg_to_user('The earliest purchase date is 3/2/2022.')
</action>

Workflow 3: Browse products in a specific (sub-)category
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Home & Kitchen" section, which likely contains the Cabinets, Racks & Shelves category. I will start by hovering over the "Home & Kitchen" menu item to reveal the subcategories.
</think>
<action>
hover('home_kitchen_id')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Storage & Organization" section under the "Home & Kitchen" menu. I will start by hovering over the "Storage & Organization" menu item to reveal the subcategories.
</think>
<action>
hover('storage_organization_id')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to click on the "Racks, Shelves & Drawers" link under the "Storage & Organization" menu. This will take me to the desired category.
</think>
<action>
click('racks_shelves_drawers_id')
</action>
"""