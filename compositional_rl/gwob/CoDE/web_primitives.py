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

"""Web design primitives and transitions."""

# placeholders for primitives and transitions
PAGE_PH = '#PAGE#'
SOURCE_PH = '#SOURCEID'
TARGET_PH = '#TARGETID'
PRECONDITION_PH = '#PRECONDITIONID'

# Concept names to primitive description mapping.
# This mapping is used to access primitives from design actions and test
# environments.
CONCEPTS2DESIGN = {
    'navbar':
        '{"concept": "navbar", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"menuItems": ["Home", "Login", "Account", "Cart", '
        '"Checkout"], "endOnClick": true}}',
    'carousel':
        '{"concept": "carousel", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"numItems": 5, "itemNames": ["1", "2", "3", "4", "5"],'
        ' "endOnClick": true}}',
    'dealmedia':
        '{"concept": "dealmedia", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"title": "Deal of the Day", "text": "Gaming '
        'workstation", "link": "Get it today!", "endOnClick": true}}',
    'header_select_items':
        '{"concept": "header", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"headerType": 5, "headerText": "Select items", '
        '"isCardHeader": false}}',
    'deck':
        '{"concept": "deck", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"numCards": 4, "cardTitles": ["Title 1", "Title 2"], '
        '"cardText": ["Product description 1", "Product description 2"], '
        '"cardNames": ["Card 1", "Card 2"], "cardHeaders": ["$0.99", "$1.99"],'
        ' "numStars": [4, 3], "endOnClick": true}}',
    'next_login_page':
        '{"concept": "next", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"buttonText": "Login"}}',
    'header_login':
        '{"concept": "header", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"headerType": 5, "headerText": "Login", "isCardHeader": '
        'false}}',
    'username':
        '{"concept": "username", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putPlaceholder": true, "putLabel": false, "labelText": '
        '"Username"}}',
    'password':
        '{"concept": "password", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putPlaceholder": true, "putLabel": false, "labelText": '
        '"Password"}}',
    'rememberme':
        '{"concept": "rememberme", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putLabel": true, "labelText": "Remember me"}}',
    'captcha':
        '{"concept": "captcha", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putLabel": true, "labelText": "Enter Captcha"}}',
    'stayloggedin':
        '{"concept": "stayloggedin", "is_core_concept": true, "source_page": '
        '' + PAGE_PH +
        ', "controls": {"putLabel": true, "labelText": "Stay logged in"}}',
    'forgotusername':
        '{"concept": "forgotusername", "is_core_concept": false, '
        '"source_page": ' + PAGE_PH +
        ', "controls": {"text": "Forgot user name.", "endOnClick": true}}',
    'forgotpassowrd':
        '{"concept": "forgotpassword", "is_core_concept": false, '
        '"source_page": ' + PAGE_PH +
        ', "controls": {"text": "Forgot password.", "endOnClick": true}}',
    'next_login':
        '{"concept": "next", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"buttonText": "Login and Checkout"}}',
    'cart':
        '{"concept": "cart", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"wrapInCard": true, "numItems": 3, "itemNames": ["Shoe",'
        ' "Bag", "Tshirt"], "endOnClick": true}}',
    'next_checkout':
        '{"concept": "next", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"buttonText": "Checkout"}}',
    'header':
        '{"concept": "header", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"headerType": 5, "headerText": "Shipping Information", '
        '"isCardHeader": false}}',
    'firstname':
        '{"concept": "name_first", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putPlaceholder": false, "putLabel": true, "labelText": '
        '"First Name"}}',
    'lastname':
        '{"concept": "name_last", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putPlaceholder": false, "putLabel": true, "labelText": '
        '"Last Name"}}',
    'addressline1':
        '{"concept": "address_line1", "is_core_concept": true, "source_page": '
        + PAGE_PH + ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Address"}}',
    'addressline2':
        '{"concept": "address_line2", "is_core_concept": true, "source_page": '
        + PAGE_PH + ', "controls": {"putPlaceholder": true, "putLabel": false, '
        '"labelText": ""}}',
    'city':
        '{"concept": "city", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putPlaceholder": false, "putLabel": true, "labelText": '
        '"City"}}',
    'zipcode':
        '{"concept": "postal_code", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putPlaceholder": false, "putLabel": true, "labelText": '
        '"ZIP Code"}}',
    'state':
        '{"concept": "state", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"putLabel": false, "labelText": "State", "values": ["CA",'
        ' "NY"]}}',
    'submit':
        '{"concept": "submit", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', '
        '"controls": {"buttonText": "Place Order"}}',
    'footer1':
        '{"concept": "footer", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"footerItems": ["Contact", "Terms", "Support", "Full '
        'Site"], "endOnClick": true}}',
    'inpgroup1':
        '{"concept": "inputgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"putPlaceholder": true, "putLabel": false, '
        '"labelText": "Search"}}',
    '#none#':
        None
}

# Additional mappings for flight booking domain.
CONCEPTS2DESIGN_FLIGHT = {
    'departureairport':
        '{"concept": "departureairport", "is_core_concept": true, "source_page": '
        + PAGE_PH + ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "From"}}',
    'destinationairport':
        '{"concept": "destinationairport", "is_core_concept": true, '
        '"source_page": ' + PAGE_PH +
        ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "To"}}',
    'departuredate':
        '{"concept": "departuredate", "is_core_concept": true, "source_page": '
        + PAGE_PH + ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Depart"}}',
    'destinationdate':
        '{"concept": "destinationdate", "is_core_concept": true, "source_page": '
        + PAGE_PH + ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Return"}}',
    'numberofpeople':
        '{"concept": "numberofpeople", "is_core_concept": true, "source_page": '
        + PAGE_PH + ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Number of passengers"}}',
    'cabin':
        '{"concept": "cabin", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', "controls": {"name": "cabin", "header": "Cabin", "items": '
        '["Economy", "First"]}}',
    'flighttype':
        '{"concept": "flighttype", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', "controls": {"name": "flighttype", "header": "", "items": '
        '["Oneway", "Roundtrip"]}}'
}

# Additional mappings for payment domain.
CONCEPTS2DESIGN_PAYMENT = {
    'cc':
        '{"concept": "credit_card_type", "is_core_concept": true, "source_page": '
        + PAGE_PH +
        ', "controls": {"header": "Payment", "items": ["Credit Card", "Debit '
        'Card"]}}',
    'fullname':
        '{"concept": "name_full", "is_core_concept": true, "source_page": ' +
        PAGE_PH + ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Full Name"}}',
    'ccnumber':
        '{"concept": "credit_card_number", "is_core_concept": true, '
        '"source_page": ' + PAGE_PH +
        ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Credit card number"}}',
    'ccexpdate':
        '{"concept": "credit_card_expiration", "is_core_concept": true, '
        '"source_page": ' + PAGE_PH +
        ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "Expiration date"}}',
    'cccvv':
        '{"concept": "credit_card_verification_code", "is_core_concept": true,'
        ' "source_page": ' + PAGE_PH +
        ', "controls": {"putPlaceholder": false, "putLabel": true, '
        '"labelText": "CVV"}}'
}

# TODO(izzeddin): passive is not currently used in the trained models
# Additional mapping for some passive or auxiliary primitives.
CONCEPTS2DESIGN_PASSIVE = {
    'navbar2':
        '{"concept": "navbar", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"homeLink": "GVideo", "menuItems": ["Home", '
        '"Trending", "Subscriptions", "Library"], "endOnClick": true}}',
    'navbar3':
        '{"concept": "navbar", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"homeLink": "GTube", "menuItems": ["Browse", "Music", '
        '"Video", "Settings"], "endOnClick": true}}',
    'inpgroup2':
        '{"concept": "inputgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"putPlaceholder": true, "putLabel": false, '
        '"labelText": "Enter keywords"}}',
    'inpgroup3':
        '{"concept": "inputgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"putPlaceholder": true, "putLabel": false, '
        '"labelText": "Find"}}',
    'inpgroup4':
        '{"concept": "inputgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"putPlaceholder": true, "putLabel": false, '
        '"labelText": "Search video"}}',
    'footer2':
        '{"concept": "footer", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"footerItems": ["Company", "Work with Us", "Our Team",'
        ' "Legal"], "endOnClick": true}}',
    'footer3':
        '{"concept": "footer", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"footerItems": ["Follow use on Twitter", "Follow use '
        'on Facebook", "Email Us"], "endOnClick": true}}',
    'link1':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"text": "Gift Cards", "endOnClick": true}}',
    'link2':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"text": "Redeem", "endOnClick": true}}',
    'link3':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"text": "For developers", "endOnClick": true}}',
    'link4':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"text": "Customer Service", "endOnClick": true}}',
    'link5':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"text": "Give Feedback", "endOnClick": true}}',
    'link6':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH + ', "controls": {"text": "Top Deals", "endOnClick": true}}',
    'link7':
        '{"concept": "linkgroup", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"text": "Saved Searches", "endOnClick": true}}',
    'radio1':
        '{"concept": "singleselect", "is_core_concept": false, "source_page": '
        + PAGE_PH +
        ', "controls": {"name": "pickup", "header": "", "items": ["Pickup at '
        'Store", "Ship"]}}',
    'radio2':
        '{"concept": "singleselect", "is_core_concept": false, "source_page": '
        + PAGE_PH +
        ', "controls": {"name": "itemsize", "header": "Select Size", "items": '
        '["S", "M", "L"]}}',
    'media1':
        '{"concept": "dealmedia", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"title": "Latest news on cloud computing", "text": '
        '"New developments on cloud computing.", "link": "Read the Full '
        'Article.", "endOnClick": true}}',
    'media2':
        '{"concept": "dealmedia", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"title": "Learn more about neural networks.", "text": '
        '"Recent technological advances in neural networks.", "link": "Go to '
        'the News.", "endOnClick": true}}',
    'media3':
        '{"concept": "dealmedia", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"title": "5 Most Downloaded Apps from Store.", "text":'
        ' "Learn more about our top 5 choices this year.", "link": "View Full '
        'Coverage.", "endOnClick": true}}',
    'media4':
        '{"concept": "dealmedia", "is_core_concept": false, "source_page": ' +
        PAGE_PH +
        ', "controls": {"title": "New Battle Between PS and XBox Frontend.", '
        '"text": "See how both companies are pushing for the earliest '
        'release.", "link": "Find More about It.", "endOnClick": true}}'
}

CONCEPTS2DESIGN.update(CONCEPTS2DESIGN_PAYMENT)
CONCEPTS2DESIGN.update(CONCEPTS2DESIGN_FLIGHT)

# These are predefined transitions. If transitions are not learned, these
# can be added by default to the website.
PREDEFINED_TRANSITIONS2DESIGN = {
    'addOpenPageTransition1':
        '{"type": "addOpenPageTransition", "is_transition": true, '
        '"source_group": "group_next_p0", "target_group": "page1", '
        '"target_page": 1, "controls": {"eventType": "click", "conceptual": '
        'false, "shouldSubmitOnFinalPage": true, "taskSuccessScale": 1.0, '
        '"preconditionVisited": []}}',
    'addOpenPageTransition2':
        '{"type": "addOpenPageTransition", "is_transition": true, '
        '"source_group": "group_next_p1", "target_group": "page2", '
        '"target_page": 2, "controls": {"eventType": "click", "conceptual": '
        'false, "shouldSubmitOnFinalPage": true, "taskSuccessScale": 1.0, '
        '"preconditionVisited": []}}',
    'addOpenPageTransition3':
        '{"type": "addOpenPageTransition", "is_transition": true, '
        '"source_group": "group_next_p2", "target_group": "page3", '
        '"target_page": 3, "controls": {"eventType": "click", "conceptual": '
        'false, "shouldSubmitOnFinalPage": true, "taskSuccessScale": 1.0, '
        '"preconditionVisited": []}}',
    'addSubmitTransition':
        '{"type": "addSubmitTransition", "is_transition": true, '
        '"source_group": "group_submit_p3", "target_page": 3, "controls": '
        '{"conceptual": false, "taskSuccessScale": 1.0, "preconditionVisited":'
        ' []}}',
}

# Transition names to transition description mapping. These can be used to add
# transitions to web elements that enables complex dynamics.
TRANSITIONS2DESIGN = {
    'addShowHideTransition':
        '{"type": "addShowHideTransition", "is_transition": true, '
        '"source_group": "' + SOURCE_PH + '", "target_group": "' + TARGET_PH +
        '", '
        '"controls": {"eventType": "keypress", "flipEvent": false}}',
    'addOpenPageTransition':
        '{"type": "addOpenPageTransition", "is_transition": true, '
        '"source_group": "' + SOURCE_PH + '", "target_group": "' + TARGET_PH +
        '", '
        '"target_page": -1, "controls": {"eventType": "click", "conceptual": '
        'false, "shouldSubmitOnFinalPage": true, "taskSuccessScale": 1.0, '
        '"preconditionVisited": [' + PRECONDITION_PH + ']}}',
    'addSubmitTransition':
        '{"type": "addSubmitTransition", "is_transition": true, '
        '"source_group": "' + SOURCE_PH + '", "target_page": -1, "controls": '
        '{"conceptual": false, "taskSuccessScale": 1.0, "preconditionVisited":'
        ' [' + PRECONDITION_PH + ']}}',
}

# Ordered list of primitives and transitions.
SETOFTRANSITIONS = sorted(TRANSITIONS2DESIGN.keys())
CONCEPTS = sorted(CONCEPTS2DESIGN.keys())
