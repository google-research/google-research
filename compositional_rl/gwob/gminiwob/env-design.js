// Copyright 2022 The Google Research Authors.
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

/**
 * @fileoverview Definitions of primitives, transitions, low-level navigation.
 *
 * We use concept and primitive interchangeably in this file.
 */

// Make sure to import jquery and miniwob/html/core before importing this file.

// Common primitives.
const usernameConcept = 'username';
const passwordConcept = 'password';
const firstnameConcept = 'name_first';
const lastnameConcept = 'name_last';
const addressline1Concept = 'address_line1';
const addressline2Concept = 'address_line2';
const cityConcept = 'city';
const stateConcept = 'state';
const zipcodeConcept = 'postal_code';

// Payment primitives
const fullnameConcept = 'name_full';
const cctypeConcept = 'credit_card_type';
const ccnumberConcept = 'credit_card_number';
const ccexpirationdateConcept = 'credit_card_expiration';
const cccvvConcept = 'credit_card_verification_code';

// main primitives
const navbarConcept = 'navbar';
const carouselConcept = 'carousel';
const deckConcept = 'deck';
const dealmediaConcept = 'dealmedia';
const headerConcept = 'header';
const captchaConcept = 'captcha';
const remembermeConcept = 'rememberme';
const stayloggedinConcept = 'stayloggedin';
const submitConcept = 'submit';
const cartConcept = 'cart';
const footerConcept = 'footer';
const inputgroupConcept = 'inputgroup';

// login primitives
const nextConcept = 'next';
const forgotusernameConcept = 'forgotusername';
const forgotpasswordConcept = 'forgotpassword';

// Flight booking primitives
const departureairportConcept = 'departureairport';
const destinationairportConcept = 'destinationairport';
const departuredateConcept = 'departuredate';
const destinationdateConcept = 'destinationdate';
const numberofpeopleConcept = 'numberofpeople';
const flighttypeConcept = 'flighttype';
const cabinConcept = 'cabin';

// Primitive to input type mapping
const concept2InputType = new Map();
concept2InputType.set(passwordConcept, 'password');
concept2InputType.set(stayloggedinConcept, 'checkbox');
concept2InputType.set(remembermeConcept, 'checkbox');
concept2InputType.set(submitConcept, 'submit');
concept2InputType.set(stateConcept, 'select');
concept2InputType.set(numberofpeopleConcept, 'number');

REGISTERED_CONCEPTS = {};

/**
********************************************************************************
* Common label, text, and IDs for groups, inputs, labels, links, and modals
********************************************************************************
*/

/**
 * List of concepts that have binary profiles.
 * @return {!Array<number>}
 */
function getBinaryConcepts() {
  return [
    stayloggedinConcept, remembermeConcept
  ];
}

/**
 * Label to be used for the concept. For now, there is only one label per
 * concept but this can be increased by adding new items to lists.
 * @param {string} concept A concept name
 * @param {string=} defaultValue The default value for the concept if not found
 * @return {string} A label for the concept to be used in html
 */
function getLabelForConcept(concept, defaultValue = null) {
  const submitConceptLabels =
      ['Submit'];  //, 'Go', 'Finish', 'Enter', 'Confirm'];
  switch (concept) {
    case firstnameConcept:
      return core.sample(
          ['First Name']);  //, 'Initial Name', 'Given Name', 'Forename'])
    case lastnameConcept:
      return core.sample(['Last Name']);
    case usernameConcept:
      return core.sample(['Username']);
    case passwordConcept:
      return core.sample(
          ['Password']);  //, 'Watchword', 'Word', 'Parole', 'Countersign'])
    case submitConcept:
      return core.sample(submitConceptLabels);
    case nextConcept:
      return core.sample(
          ['Next']);  //, 'Continue', 'Forward'].concat(submitConceptLabels));
    case captchaConcept:
      return core.sample(
          ['Enter Captcha']);  //, 'Security Keywords', 'Enter Code']);
    case navbarConcept:
      return core.sample(['Navigation bar.']);
    case stayloggedinConcept:
      return core.sample(['Stay logged in.']);
    case remembermeConcept:
      return core.sample(['Remember me.']);
    case forgotusernameConcept:
      return core.sample(['Forgot username.']);
    case forgotpasswordConcept:
      return core.sample(['Forgot password.']);
    case cartConcept:
      return core.sample(['Cart']);
    case addressline1Concept:
      return core.sample(['Address']);
    case addressline2Concept:
      return core.sample(['Apt #']);
    case cityConcept:
      return core.sample(['City']);
    case zipcodeConcept:
      return core.sample(['Zipcode']);
    case stateConcept:
      return core.sample(['State']);
    case dealmediaConcept:
      return core.sample(['Deal of the Day']);
    case footerConcept:
      return core.sample(['Contact']);
    case departureairportConcept:
      return core.sample(['From']);
    case destinationairportConcept:
      return core.sample(['To']);
    case departuredateConcept:
      return core.sample(['Depart']);
    case destinationdateConcept:
      return core.sample(['Return']);
    case numberofpeopleConcept:
      return core.sample(['Number of people']);
    case flighttypeConcept:
      return core.sample(['Flight type']);
    case cabinConcept:
      return core.sample(['Cabin selection']);
  }
  return defaultValue;
}

/**
 * Error text to be used in the concept.
 * @param {string} concept A concept name
 * @return {string} An error text to use in html
 */
function getErrorForConcept(concept) {
  const errorObj = {};
  errorObj.text = 'input';
  switch (concept) {
    case firstnameConcept:
      errorObj.text =
          core.sample(['first name'])
              .toLowerCase();  //, 'Initial Name', 'Given Name', 'Forename'])
                               // .toLowerCase();
      break;
    case usernameConcept:
      errorObj.text =
          core.sample(['username'])
              .toLowerCase();  //, 'Initial Name', 'Given Name', 'Forename'])
                               // .toLowerCase();
      break;
    case passwordConcept:
      errorObj.text =
          core.sample(['password']).toLowerCase();  //, 'Watchword', 'Word',
                                                    //'Parole', 'Countersign'])
                                                    // .toLowerCase()
      break;
  }
  return core.sample([
    'Your ' + errorObj.text + ' is invalid',
    'Valid ' + errorObj.text + ' is required.'
  ]);
}

/**
 * Input types for the concept.
 * @param {string} concept A concept name
 * @return {string} Input type for the corresponding concept
 */
function getInputTypeForConcept(concept) {
  console.log(concept2InputType);
  console.log([concept2InputType.has(concept), concept]);
  if (concept2InputType.has(concept)) {
    return concept2InputType.get(concept);
  }
  return 'text';
}

/**
 * Generates an html id that corresponds to the concept.
 * These ids will be used to take actions and compute reward.
 * Ids are always global but they might reflect their corresponding page and
 * a unique postfix as well. For example, if firstnameConcept is a core
 * concept that will be accessed globally, its id is independent of the page.
 * However, a navbar might appear in multiple pages and it contains multiple
 * items. So, each item will be of the form
 * group_navbar_p{pageIndex}_i{uniqueOrder} where pageIndex is the number of
 * the page and uniqueOrder is another number that uniquely orders similar
 * elements.
 * @param {string} concept A concept name
 * @param {number=} pageIndex Index of the page to generate an html id
 * @param {number=} unqIndex A unique index to append to the html id
 * @return {string} An html id that corresponds to the concept
 */
function getGroupIdForConcept(concept, pageIndex = null, unqIndex = null) {
  const idObj = {};
  idObj.id = `group_${concept}`;
  if (pageIndex != null && pageIndex >= 0)
    idObj.id = idObj.id.concat(`_p${pageIndex}`);
  if (unqIndex != null) idObj.id = idObj.id.concat(`_i${unqIndex}`);
  return idObj.id;
}

/**
 * Similar to group ids above, generates an html id for those element that
 * can be interacted with via clicking or using keyboard. These actionable
 * elements can only appear as a descendant of a group element. Think of the
 * group elements as a high level semantic group and actionable elements
 * are those elements that we can interact within the same semantics.
 * @param {string} concept A concept name
 * @param {number=} pageIndex Index of the page to generate an html id
 * @param {number=} unqIndex A unique index to append to the html id
 * @return {string} An html id that corresponds to an actionable element under
 *     the given concept
 */
function getActionableIdForConcept(concept, pageIndex = null, unqIndex = null) {
  const idObj = {};
  idObj.id = `actionable_${concept}`;
  if (pageIndex != null && pageIndex >= 0)
    idObj.id = idObj.id.concat(`_p${pageIndex}`);
  if (unqIndex != null) idObj.id = idObj.id.concat(`_i${unqIndex}`);
  return idObj.id;
}

/**
 * Similar to above, generates an id for a label element. Usually, this id
 * is not used to access the element by added for might-need-in-the-future
 * bases.
 * @param {string} concept A concept name
 * @param {number=} pageIndex Index of the page to generate an html id
 * @param {number=} unqIndex A unique index to append to the html id
 * @return {string} An html id that corresponds to a label under the given
 *     concept
 */
function getLabelIdForConcept(concept, pageIndex = null, unqIndex = null) {
  const idObj = {};
  idObj.id = `label_${concept}`;
  if (pageIndex != null && pageIndex >= 0)
    idObj.id = idObj.id.concat(`_p${pageIndex}`);
  if (unqIndex != null) idObj.id = idObj.id.concat(`_i${unqIndex}`);
  return idObj.id;
}

/**
********************************************************************************
# Register and sample a concept to use in page design
********************************************************************************
*/
/**
 * Register your primitive object in a global dictionary to sample from.
 * There might be multiple different primitives that correspond to the same
 * semantics. For example, you might implement a navigation bar in two
 * different libraries. By registering them, you have the option of sampling
 * from the two and randomizing your website look.
 * If your concept is independent of a page and unique across websites,
 * use sourcePage=''.
 * @param {string} concept A concept name
 * @param {number} sourcePage Index of the page to register the primitive
 * @param {!ObjType} jqObj A unique index to append to the html id
 */
function registerConcept(concept, sourcePage, jqObj) {
  const conceptName = `${concept}_${sourcePage}`;
  if (!REGISTERED_CONCEPTS.hasOwnProperty(conceptName))
    REGISTERED_CONCEPTS[conceptName] = [];
  REGISTERED_CONCEPTS[conceptName].push(jqObj);
}

/**
 * Sample a registered jquery object corresponding to your concept.
 * Similar to registering, use sourcePage='' if your concept is global and
 * independent of a specific page.
 * @param {string} concept A concept name
 * @param {number} sourcePage Index of the page to register the primitive
 * @return {!ObjType} A jquery object sampled from the registered primitives
 */
function sampleConcept(concept, sourcePage) {
  const conceptName = `${concept}_${sourcePage}`;
  if (!REGISTERED_CONCEPTS.hasOwnProperty(conceptName)) return null;
  return core.sample(REGISTERED_CONCEPTS[conceptName]);
}

/**
********************************************************************************
* Auxiliary functions
********************************************************************************
*/
/**
 * Fill the input array by cycling through its elements and reinserting.
 * @param {!Array<!ObjType|number|string>} array An array
 * @param {number} size Size of the final array
 * @return {!Array<!ObjType|number|string>} A new array of given size filled by
 *     repeating the input array
 */
function fillArray(array, size) {
  if (array.length >= size) return array;
  let newArray = array;
  const l = newArray.length;
  for (let i = newArray.length; i < size; i++) {
    newArray.push(array[i % l]);
  }
  return newArray;
}

/**
 * A function with a constant output list that returns bootstrap 'card' names.
 * @return {!Array<string>} An array of class names for card primitive
 */
function getCardClassNames() {
  card = 'card h-100';
  cardBody = 'card-body';
  cardTitle = 'card-title';
  cardText = 'card-text';
  cardFooter = 'card-footer';
  return [card, cardBody, cardTitle, cardText, cardFooter];
}

/**
 * Find the element corresponding to its id and optionally check if it is
 * visible. Then, find the ancestor that corresponds to the page and fetch
 * the page number from it.
 * @param {string} id An html id
 * @param {boolean=} checkVisible A boolean to check if the element is visible
 * @return {number} Index of the corresponding page
 */
function getPageNumber(id, checkVisible = false) {
  let pageIndex = -1;
  const groups = $(`#${id}`);
  const page = groups.parentsUntil(`#area`).last().filter(function() {
    return $(this).is(':visible') || !checkVisible;
  });
  if (page.length > 0)
    pageIndex = parseInt(page.attr('id').substring('page'.length));
  return pageIndex;
}

/**
********************************************************************************
* Main functions for creating conceptual groups
********************************************************************************
*/
// Page designs including landing page and form submissions.

/**
 * Adds pages to the parent with a single form in each page. If we have more
 * than 1 and each form is unempty, pages should be connected. pageType
 * denotes if the page is a core page, i.e., it should be visited to solve a
 * task. Usually, most pages are core pages.
 * @param {!ObjType} parent A jquery object
 * @param {number} numPages An integer number of pages
 * @param {string} pageType A string of page type
 * @return {!ObjType} A jquery object where empty pages are added
 */
function _appendEmptyPagesWithForms(parent, numPages, pageType) {
  for (let i = 0; i < numPages; i++) {
    $(`#page${i}`).hide();
    if (i > 0) {  // these pages are hidden
      parent.append(`<div id="page${i}" style="display:none;" class="${
          pageType}"> <div class="container"> <div class="row"> <div class="col-md-12 order-md-1"><div id="mainform"></div></div></div></div> </div>`);
    } else {
      parent.append(`<div id="page${i}" class="${
          pageType}"> <div class="container"> <div class="row"> <div class="col-md-12 order-md-1"><div id="mainform"></div></div></div></div> </div>`);
    }
  }
  return parent;
}

/**
 * Creates and registers a conceptual navbar primitive.
 * controls=(menuItems,homeLink,endOnClick)
 * actionable ids=(actionable_{concept}_{i})
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addNavbar(sourcePage, concept, controls = null) {
  // default values
  let menuItems = ['Home', 'About', 'Contact'];
  let homeLink = 'GShopping';

  // setup from input controls
  if (controls && controls.hasOwnProperty('menuItems')) {
    menuItems = controls.menuItems;
  }
  if (controls && controls.hasOwnProperty('homeLink')) {
    homeLink = controls.homeLink;
  }

  // main html
  let menuItemsObj =
      $(`<div class="collapse navbar-collapse" id="navbarResponsive${
          sourcePage}"></div>`);
  let menuItemsLiObj = $(`<ul class="navbar-nav ml-auto"></ul>`);
  let groupConceptIndex = 0, actionableConceptIndex = 0;
  for (let i = 0; i < menuItems.length; i++) {
    const menuItemObj =
        $(`<li id="${
              getGroupIdForConcept(
                  concept, sourcePage, groupConceptIndex++)}" name="${
              concept}-menu-${menuItems[i]}" class="nav-item"></li>`)
            .append(`<a id="${
                getActionableIdForConcept(
                    concept, sourcePage,
                    actionableConceptIndex++)}" class="nav-link" href="#">${
                menuItems[i]}</a>`);
    menuItemsLiObj.append(menuItemObj);
  }
  menuItemsObj.append(menuItemsLiObj);
  let navbarContainer = $(`<div class="container"></div>`);
  let navbarBrand = $(`<div id="${
      getGroupIdForConcept(
          concept, sourcePage,
          groupConceptIndex++)}" name="${concept}-brand-name" > <a id="${
      getActionableIdForConcept(
          concept, sourcePage,
          actionableConceptIndex++)}" class="navbar-brand" href="#">${
      homeLink}</a> </div>`);
  let navbarButton = $(`<div id="${
      getGroupIdForConcept(
          concept, sourcePage,
          groupConceptIndex++)}" name="${concept}-menu" > <button id="${
      getActionableIdForConcept(
          concept, sourcePage,
          actionableConceptIndex++)}" class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarResponsive${
      sourcePage}" aria-controls="navbarResponsive${
      sourcePage}" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button> </div>`);
  navbarContainer.append(navbarBrand, [navbarButton, menuItemsObj]);
  let navbar = $(`<nav class="navbar navbar-dark bg-dark"></nav>`)
                   .append(navbarContainer);

  // If end on click, actionable elements will terminate the episode when
  // clicked
  if (controls.endOnClick) {
    menuItemsLiObj.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
    navbarBrand.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
  }
  registerConcept(concept, sourcePage, navbar);
}

/**
 * Creates and registers a conceptual carousel primitive.
 *  controls=(numItems,itemNames,endOnClick)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addCarousel(sourcePage, concept, controls = null) {
  // default values
  let numItems = 3;
  let itemNames = ['First', 'Second', 'Third'];

  // setup from input controls
  if (controls && controls.hasOwnProperty('numItems')) {
    numItems = controls.numItems;
    for (let i = 3; i < numItems; i++) {
      itemNames.push(`${i + 1}th`);
    }
  }
  if (controls && controls.hasOwnProperty('itemNames')) {
    itemNames = controls.itemNames;
  }

  // fill controls by repeating
  itemNames = fillArray(itemNames, numItems);

  // main html
  let carousel = $(
      `<div id="carouselExampleIndicators" class="carousel slide my-4" data-interval="false" data-ride="carousel"></div>`);
  let carouselIndicators = $(`<ol class="carousel-indicators"></ol>`);
  let carouselInner = $(`<div class="carousel-inner" role="listbox"></div>`);
  for (let i = 0; i < numItems; i++) {
    if (i == 0) {
      carouselIndicators.append(
          `<li data-target="#carouselExampleIndicators" data-slide-to="${
              i}" class="active"></li>`);
      carouselInner.append(`<div id="${
          getGroupIdForConcept(concept, sourcePage, i)}" name="${
          concept}-carousel-item${i}" class="carousel-item  active">
              <img id="${
          getActionableIdForConcept(
              concept, sourcePage,
              i)}" class="d-block img-fluid" src="images/900x350.png" alt="${
          itemNames[i]} slide">
            </div>`);
    } else {
      carouselIndicators.append(
          `<li data-target="#carouselExampleIndicators" data-slide-to="${
              i}" class=""></li>`);
      carouselInner.append(`<div id="${
          getGroupIdForConcept(
              concept, sourcePage,
              i)}" name="${concept}-carousel-item${i}" class="carousel-item">
              <img id="${
          getActionableIdForConcept(
              concept, sourcePage,
              i)}" class="d-block img-fluid" src="images/900x350.png" alt="${
          itemNames[i]} slide">
            </div>`);
    }
  }
  let prevButton = $(`<div id="${
      getGroupIdForConcept(
          concept, sourcePage,
          numItems)}" name="${concept}-carousel-backward"><a id="${
      getActionableIdForConcept(
          concept, sourcePage,
          numItems)}" class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">
            <span class="carousel-control-prev-icon" aria-hidden="true"></span>
            <span class="sr-only">Previous</span>
          </a></div>`);
  let nextButton = $(`<div id="${
      getGroupIdForConcept(
          concept, sourcePage,
          numItems + 1)}" name="${concept}-carousel-forward"><a id="${
      getActionableIdForConcept(
          concept, sourcePage,
          numItems +
              1)}" class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">
            <span class="carousel-control-next-icon" aria-hidden="true"></span>
            <span class="sr-only">Next</span>
          </a></div>`);
  carousel.append(carouselIndicators, [carouselInner, prevButton, nextButton]);

  // If end on click, actionable elements will terminate the episode when
  // clicked
  if (controls.endOnClick) {
    carouselInner.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
    prevButton.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
    nextButton.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
  }

  registerConcept(concept, sourcePage, carousel);
}

/**
 * Creates and registers a conceptual deck primitive.
 * controls=(numCards,cardNames,cardText,cardTitles,cardHeaders,numStars,endOnClick)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addDeck(sourcePage, concept, controls = null) {
  // default values
  let numCards = 3;
  let cardNames = ['First Card', 'Second Card', 'Third Card'];
  let cardText = [
    'Lorem ipsum dolor sit amet, consectetur adipisicing elit. Amet numquam aspernatur!'
  ];
  let cardTitles = ['Item One', 'Item Two', 'Item Three'];
  let cardHeaders = ['$0.99', '$1.99', '$2.99'];
  let numStars = [5, 4, 3];

  // setup from input controls
  if (controls && controls.hasOwnProperty('numCards')) {
    numCards = controls.numCards;
    for (let i = 3; i < numCards; i++) {
      cardNames.push(`${i + 1}th`);
    }
  }
  if (controls && controls.hasOwnProperty('cardNames')) {
    cardNames = controls.cardNames;
  }
  if (controls && controls.hasOwnProperty('cardText')) {
    cardText = controls.cardText;
  }
  if (controls && controls.hasOwnProperty('cardTitles')) {
    cardTitles = controls.cardTitles;
  }
  if (controls && controls.hasOwnProperty('cardHeaders')) {
    cardHeaders = controls.cardHeaders;
  }
  if (controls && controls.hasOwnProperty('numStars')) {
    numStars = controls.numStars;
  }

  // fill controls by repeating
  cardNames = fillArray(cardNames, numCards);
  cardText = fillArray(cardText, numCards);
  cardTitles = fillArray(cardTitles, numCards);
  cardHeaders = fillArray(cardHeaders, numCards);
  numStars = fillArray(numStars, numCards);

  // main html
  function getStars(starNumber) {
    let stars = ``;
    for (let i = 0; i < 5; i++) {
      if (i < starNumber) {
        stars += ` &#9733;`;
      } else {
        stars += ` &#9734;`;
      }
    }
    return stars;
  }
  let deck = $(`<div class="container-fluid"></div>`);
  for (let i = 0; i < numCards; i++) {
    deck.append(`<div class="row">
          <div class="col">
            <div class="card h-100">
              <div id="${
        getGroupIdForConcept(
            concept, sourcePage,
            2 * i)}" name="${concept}-deck-img${i}"><a id="${
        getActionableIdForConcept(
            concept, sourcePage,
            2 * i)}" href="#"><img class="card-img-top" src="images/700x400.png" alt="${
        cardNames[i]}"></a></div>
              <div class="card-body">
                <h4 class="card-title">
                  <div id="${
        getGroupIdForConcept(
            concept, sourcePage,
            2 * i + 1)}" name="${concept}-deck-link${i}"><a id="${
        getActionableIdForConcept(
            concept, sourcePage,
            2 * i + 1)}" href="#">${cardTitles[i]}</a></div>
                </h4>
                <h5>${cardHeaders[i]}</h5>
                <p class="card-text">${cardText[i]}</p>
              </div>
              <div class="card-footer">
                <small class="text-muted">${getStars(numStars[i])}</small>
              </div>
            </div>
          </div>
          </div>`);
  }

  if (controls.endOnClick) {
    deck.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
  }

  registerConcept(concept, sourcePage, deck);
}

/**
 * Creates and registers a conceptual header primitive.
 * controls=(headerType,headerText,isCardHeader)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addHeader(sourcePage, concept, controls = null) {
  // default values
  let headerText = 'Header';
  let headerType = '1';
  let className = '';

  // setup from input controls
  if (controls && controls.hasOwnProperty('headerText')) {
    headerText = controls.headerText;
  }
  if (controls && controls.hasOwnProperty('headerType')) {
    headerType = controls.headerType;
  }
  if (controls && controls.hasOwnProperty('isCardHeader') &&
      controls.isCardHeader) {
    className = 'card-header';
  }

  // main html
  let header =
      $(`<h${headerType} class="${className}">${headerText}</h${headerType}>`);
  let group = $(`<div name="${concept}" id="${
      getGroupIdForConcept(concept, -1)}" tabindex=0></div>`);
  group.append(header);
  registerConcept(concept, sourcePage, group);
}

/**
 * Creates and registers a conceptual input group primitive.
 * If the primitive will have a corresponding profile and needed in
 * reward estimation, it is a core primitive and isCore should be true.
 * Note that not all primitives can be a core primitive.
 * controls=(putPlaceholder,putLabel,labelText,values)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {boolean} isCore If the primitive is a core primitive
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addInputGroup(sourcePage, concept, isCore, controls = null) {
  // default values
  let putLabel = false, putPlaceholder = false,
      labelText = getLabelForConcept(concept), placeholder = '', values = [''];
  let classNameForm = 'form-group';
  let classNameInput = 'form-control';
  let classNameText = 'form-text';
  let classNameLabel = '';
  if (getInputTypeForConcept(concept) ==
      'checkbox') {  // checkbox is rendered differently
    classNameForm = 'form-check';
    classNameInput = 'form-check-input';
    classNameText = 'form-check-label';
    classNameLabel = 'form-check-label';
  }

  // setup from input controls
  if (controls && controls.hasOwnProperty('putLabel')) {
    putLabel = controls.putLabel;
  }
  if (controls && controls.hasOwnProperty('putPlaceholder')) {
    putPlaceholder = controls.putPlaceholder;
  }
  if (controls && controls.hasOwnProperty('labelText')) {
    labelText = controls.labelText;
  }
  if (controls && controls.hasOwnProperty('values')) {
    values = controls.values;
  }

  // main html
  let label = null;
  let input = null;
  if (putLabel) {
    label = `<label id="${
        getLabelIdForConcept(concept, isCore ? -1 : sourcePage)}" for="${
        getActionableIdForConcept(concept, isCore ? -1 : sourcePage)}" class="${
        classNameLabel}">${labelText}</label>`;
  }
  if (putPlaceholder) {
    placeholder = getLabelForConcept(concept, labelText);
  }
  if (getInputTypeForConcept(concept) == 'select') {
    input = $(`<select id="${
        getActionableIdForConcept(
            concept,
            isCore ? -1 : sourcePage)}" class="${classNameInput}" required>`);
    for (let i = 0; i < values.length; i++) {
      input.append(`<option>${values[i % values.length]}</option>`);
    }
  } else {
    input = `<input type="${getInputTypeForConcept(concept, sourcePage)}" id="${
        getActionableIdForConcept(
            concept, isCore ? -1 : sourcePage)}" placeholder="${
        placeholder}" value="" class="${classNameInput}" required>`;
  }
  let errorTxt =
      `<div style="display:none;" class="${classNameText} text-muted">
                   ${getErrorForConcept(concept)}
                </div>`;
  // By default, div's are not focusable.
  // Not adding tabindex will cause focusing on div element to focus on
  // previously focused focusable element i.e., add tabindex=0 ! What this does
  // is that, if you try to click on a div element that doesn't have a
  // tabindex=0, it will actuall click on the previously focused element!!!
  let group = $(`<div name="${concept}" id="${
      getGroupIdForConcept(concept, isCore ? -1 : sourcePage)}" class="${
      classNameForm}" tabindex=0></div>`);
  if (getInputTypeForConcept(concept) == 'checkbox') {
    if (putLabel && labelText != '') {
      group.append(input, [label, errorTxt]);
    } else {
      group.append(input, [errorTxt]);
    }
  } else {
    if (putLabel && labelText != '') {
      group.append(label, [input, errorTxt]);
    } else {
      group.append(input, [errorTxt]);
    }
  }
  registerConcept(concept, sourcePage, group);
}


/**
 * Creates and registers a generic conceptual button.
 * controls=(buttonText,buttonSize,isBlock)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addButton(sourcePage, concept, controls = null) {
  // default values
  let buttonText = 'Click', buttonSize = 'sm', block = 'btn-block';

  // setup from input controls
  if (controls && controls.hasOwnProperty('buttonText'))
    buttonText = controls['buttonText'];
  if (controls && controls.hasOwnProperty('buttonSize'))
    buttonSize = controls['buttonSize'];
  if (controls && controls.hasOwnProperty('isBlock') && !controls['isBlock'])
    block = '';

  // main html
  let group = $(`<div id="${getGroupIdForConcept(concept, sourcePage)}" name="${
      concept}-${buttonText}"></div>`);
  let button = $(`<button id="${
      getActionableIdForConcept(
          concept, sourcePage)}" class="btn btn-secondary btn-${buttonSize} ${
      block}">${buttonText}</button>`);
  group.append(button);

  registerConcept(concept, sourcePage, group);
}

/**
 * Creates and registers a conceptual cart primitive.
 * Cost of the items in the cart will be randomized but other properties
 * such as item names, descriptions can be customized.
 * This also adds a promo text box and a submit button.
 * controls=(wrapInCard,numItems,itemNames,itemDescriptions,endOnClick)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addCart(sourcePage, concept, controls = null) {
  // default values
  let cardClassName = '', cardBodyClassName = '', cardTitleClassName = '',
      cardTextClassName = '', cardFooterClassName = '';
  let numItems = 2;
  let itemNames = [];
  let itemDescriptions = [];

  // setup from input controls
  if (controls && controls.hasOwnProperty('wrapInCard')) {
    [cardClassName, cardBodyClassName, cardTitleClassName, cardTextClassName,
     cardFooterClassName] = getCardClassNames();
  }
  if (controls && controls.hasOwnProperty('numItems')) {
    numItems = controls.numItems;
  }
  if (controls && controls.hasOwnProperty('itemNames')) {
    itemNames = controls.itemNames;
  } else {
    for (let i = 0; i < numItems; i++) itemNames.push(`Product-${i + 1}`);
  }
  if (controls && controls.hasOwnProperty('itemDescriptions')) {
    itemDescriptions = controls.itemDescriptions;
  } else {
    for (let i = 0; i < numItems; i++)
      itemDescriptions.push('Description of the product');
  }

  // main html
  let groupConceptIndex = 0, actionableConceptIndex = 0;
  let cartItems =
      $(`<ul class="list-group mb-3 col-md-12 ${cardTextClassName}"></ul>`);
  let cartTotal = 0.0;
  for (let i = 0; i < numItems; i++) {
    const cost = Math.floor(Math.random() * 50) + 1;
    cartItems.append(`
          <li id="${
        getGroupIdForConcept(
            concept, sourcePage,
            groupConceptIndex++)}" name="${concept}-cart-item${
        groupConceptIndex}" class="list-group-item d-flex justify-content-between lh-condensed">
            <div>
              <h6 class="my-0">${itemNames[i]}</h6>
              <small class="text-muted">${itemDescriptions[i]}</small>
            </div>
            <span id="${
        getGroupIdForConcept(
            concept, sourcePage,
            groupConceptIndex++)}" name="${concept}-cart-item${
        groupConceptIndex}" class="text-muted">$${cost}<br/><a id="${
        getActionableIdForConcept(
            concept, sourcePage,
            actionableConceptIndex++)}" href="#">Remove</a></span>
          </li>`);
    cartTotal += cost;
  }
  cartItems.append(`<li class="list-group-item d-flex justify-content-between">
    <span>Total (USD)</span>
    <strong>$${cartTotal}</strong>
  </li>`);
  let cartFooter = $(`<div id="${
      getGroupIdForConcept(
          concept, sourcePage,
          groupConceptIndex++)}" class="card p-2 ${cardFooterClassName}">
          <div class="input-group">
            <div id="${
      getGroupIdForConcept(concept, sourcePage, groupConceptIndex++)}">
              <input id="${
      getActionableIdForConcept(
          concept, sourcePage,
          actionableConceptIndex++)}" type="text" class="form-control" placeholder="Promo code">
            </div>
            <div id="${
      getGroupIdForConcept(
          concept, sourcePage,
          groupConceptIndex++)}" class="input-group-append">
               <button id="${
      getActionableIdForConcept(
          concept, sourcePage,
          actionableConceptIndex++)}" type="button" class="btn btn-secondary">Redeem</button>
            </div>
          </div>
        </div>`);
  let cartHeader = $(
      `<h4 class="d-flex justify-content-between align-items-center mb-3 col-md-12 ${
          cardTitleClassName}">
         <span class="text-muted">Your cart</span>
         <span class="badge badge-secondary badge-pill">${numItems}</span>
       </h4>`);
  let cartMain =
      $('<div class="row"></div>').append(cartHeader, [cartItems, cartFooter]);
  let cartBody = $(`<div clas="${cardBodyClassName}"></div>`).append(cartMain);
  let cart =
      $(`<div class="col-md-12 mb-4 ${cardClassName}"></div>`).append(cartBody);

  if (controls.endOnClick) {
    cartItems.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
    cartFooter.find(`[id^=actionable_]`).filter('button').click(function() {
      core.endEpisode(-1.0, false);
    });
  }
  registerConcept(concept, sourcePage, cart);
}

/**
 * Creates and registers a conceptual media primitive.
 * A media primitive usually has an image, a title, description, and a link.
 * controls=(title,text,link,endOnClick)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addMedia(sourcePage, concept, controls = null) {
  // default values
  let title = getLabelIdForConcept(concept),
      text = getLabelIdForConcept(concept), link = 'Go';

  // setup from input controls
  if (controls && controls.hasOwnProperty('title')) {
    title = controls['title'];
  }
  if (controls && controls.hasOwnProperty('text')) {
    text = controls['text'];
  }
  if (controls && controls.hasOwnProperty('link')) {
    link = controls['link'];
  }

  // main html
  let mediaTitle = $(`<h5 class="mt-0">`).append(title);
  let mediaText = $(`<p></p>`).append(text);
  let mediaLink =
      $(`<a id="${
            getActionableIdForConcept(
                concept, sourcePage)}" href="#" class="stretched-link"></a>`)
          .append(link);
  let mediaBody = $(`<div class="media-body"></div>`);
  mediaBody.append(mediaTitle, [mediaText, mediaLink]);
  let media = $(`<div id="${getGroupIdForConcept(concept, sourcePage)}" name="${
                    concept}-media" class="media position-relative"></div>`)
                  .append(
                      `<img src="images/150x150.png" class="mr-3" alt="${
                          getLabelIdForConcept(concept)}">`,
                      [mediaBody]);

  if (controls.endOnClick) {
    mediaLink.click(function() {
      core.endEpisode(-1.0, false);
    });
  }
  registerConcept(concept, sourcePage, media);
}

/**
 * Creates and registers a conceptual link.
 * controls = (text,endOnClick)
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addLinkGroup(sourcePage, concept, controls = null) {
  // default values
  let text = getLabelIdForConcept(concept);

  // setup from input controls
  if (controls && controls.hasOwnProperty('text')) {
    text = controls['text'];
  }

  // main html
  let link =
      $(`<a href="#" id="${getActionableIdForConcept(concept)}">${text}</a>`);
  let group = $(`<div id="${getGroupIdForConcept(concept)}" name=${concept}-${
                    text} tabindex=0></div>`)
                  .append(link);

  if (controls.endOnClick) {
    link.click(function() {
      core.endEpisode(-1.0, false);
    });
  }
  registerConcept(concept, sourcePage, group);
}


/**
 * Creates and registers a conceptual footer primitive.
 * controls=(footerItems)
 * actionable ids=(actionable_{concept}_{i})
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addFooter(sourcePage, concept, controls = null) {
  // default values
  let footerItems = ['Privacy', 'Terms', 'Support'];

  // setup from input controls
  if (controls && controls.hasOwnProperty('footerItems')) {
    footerItems = controls.footerItems;
  }

  // main html
  let companyNameObj = $(`<p class="mb-1">2020-2021 The Company.</p>`);
  let footerItemsLiObj = $(`<ul class="list-inline"></ul>`);

  let groupConceptIndex = 0, actionableConceptIndex = 0;
  for (let i = 0; i < footerItems.length; i++) {
    const footerItemObj =
        $(`<li id="${
              getGroupIdForConcept(
                  concept, sourcePage,
                  groupConceptIndex++)}" name="${concept}-footer-${
              footerItems[i]}" class="list-inline-item"></li>`)
            .append(`<a id="${
                getActionableIdForConcept(
                    concept, sourcePage, actionableConceptIndex++)}" href="#">${
                footerItems[i]}</a>`);
    footerItemsLiObj.append(footerItemObj);
  }
  let footer = $(
      `<footer class="my-5 pt-5 text-muted text-center text-small"></footer>`);
  footer.append(companyNameObj, [footerItemsLiObj]);

  if (controls.endOnClick) {
    footerItemsLiObj.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
  }
  registerConcept(concept, sourcePage, footer);
}

/**
 * Creates and registers a conceptual single selection primitive.
 * An example would be a radio button group where only one of the buttons
 * can be true at any given time.
 * controls=(items,header,name)
 * actionable ids=(actionable_{concept}_{i})
 * @param {number} sourcePage Index of the page
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function _addSingleSelectionButtonGroup(sourcePage, concept, controls = null) {
  // default values
  let items = ['First', 'Second'];
  let header = '';
  let name = 'selection';

  // setup from input controls
  if (controls && controls.hasOwnProperty('items')) {
    items = controls.items;
  }
  if (controls && controls.hasOwnProperty('header')) {
    header = controls.header;
  }
  if (controls && controls.hasOwnProperty('name')) {
    name = controls.name;
  }

  // main html
  let headerObj = $(`<h6 class="mb-3">${header}</h6>`);
  let itemsLiObj = $(`<div class="d-block my-3 ${concept}root"></div>`);

  let groupConceptIndex = 0, actionableConceptIndex = 0;
  for (let i = 0; i < items.length; i++) {
    const itemObj = $(`<div id="${
        getGroupIdForConcept(
            concept, sourcePage,
            groupConceptIndex++)}" class="custom-control custom-radio">
                <input id="${
        getActionableIdForConcept(
            concept, sourcePage, actionableConceptIndex++)}" name="${concept}${
        name}" type="radio" class="custom-control-input" value="${items[i]}">
                <label class="custom-control-label" for="${
        getActionableIdForConcept(
            concept, sourcePage,
            actionableConceptIndex - 1)}">${items[i]}</label>
              </div>`);
    itemsLiObj.append(itemObj);
  }
  let selection = $(`<div class="d-block my-3"></div>`);
  selection.append(headerObj, [itemsLiObj]);

  if (controls.endOnClick) {
    itemsLiObj.find(`[id^=actionable_]`).click(function() {
      core.endEpisode(-1.0, false);
    });
  }
  registerConcept(concept, sourcePage, selection);
}

/**
********************************************************************************
* Main functions for creating transitions.
* Source and target groups are uniquely identified by their
* (concept, pageIndex).
********************************************************************************
*/


/**
 * Creates and adds a show/hide transition.
 * This transition initializes the targetId element by hiding it no matter
 * where in the website. sourceId element might be visible or it becomes
 * visible while navigating. When the corresponding event is fired, the
 * targetId element will be visible when interacted with the sourceId element.
 * Type of the event, whether the show/hide should alternate, and any
 * elements should be interacted as a precondition can be specified via
 * controls.
 * controls=(eventType,flipEvent,preconditionVisited)
 * @param {number} sourcePage Index of the source page
 * @param {number} targetPage Index of the target page
 * @param {string} sourceId Html id of the source element
 * @param {string} targetId Html id of the target element
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addShowHideTransition(
    sourcePage, targetPage, sourceId, targetId, controls = null) {
  // setup
  let source = $(`#${sourceId}`), target = $(`#${targetId}`),
      eventType = 'click', flipEvent = true;
  if (sourceId.startsWith('group')) {
    source = source.find(`[id^=actionable_]`);
  }
  if (source.length == 0 || target.length == 0) return;
  let tagName = source.prop('tagName'), inputType = source.attr('type');
  switch (tagName) {
    case 'INPUT':
      switch (inputType) {
        case 'text':
        case 'password':
          eventType = 'keypress';
          flipEvent = false;
          break;
      }
      break;
  }

  // init
  target.hide();

  // event
  if (controls && controls.hasOwnProperty('eventType'))
    eventType = controls.eventType;
  if (controls && controls.hasOwnProperty('flipEvent'))
    flipEvent = controls.flipEvent;
  if (eventType == 'keyboard') eventType = 'keypress';

  if (controls && controls.hasOwnProperty('preconditionVisited')) {
    for (let i = 0; i < controls['preconditionVisited'].length; i++) {
      if (!source.attr('class').includes(
              `gminiwob-precondition-${controls['preconditionVisited'][i]}`))
        source.addClass(
            `gminiwob-precondition-${controls['preconditionVisited'][i]}`);
    }
  }

  if (flipEvent) {
    source.on(eventType, function() {
      if (controls && controls.hasOwnProperty('preconditionVisited')) {
        for (let i = 0; i < controls['preconditionVisited'].length; i++) {
          if (!$(`#${controls['preconditionVisited'][i]}`)
                   .attr('class')
                   .includes('visited'))
            return;
        }
      }
      if (target.is(':visible')) return target.hide();
      return target.show();
    });
  } else {
    source.one(eventType, function() {
      if (controls && controls.hasOwnProperty('preconditionVisited')) {
        for (let i = 0; i < controls['preconditionVisited'].length; i++) {
          if (!$(`#${controls['preconditionVisited'][i]}`)
                   .attr('class')
                   .includes('visited'))
            return;
        }
      }
      return target.show();
    });
  }
}

/**
 * Creates and adds a page transition.
 * Interacting with the sourceId element will open the page that contains
 * targetId element. First, the transition extracts the corresponding page
 * DOM of the sourceId and targetId elements by traversing to its ancestors.
 * Then, the transitions adds an event to the sourceId elements such that when
 * fired it will hide the source page and show the target page.
 * sourcePage and targetPage parameters are not used but they are added
 * to adhere to a standard for transitions.
 * controls=(eventType,preconditionVisited)
 * @param {number} sourcePage Index of the source page
 * @param {number} targetPage Index of the target page
 * @param {string} sourceId Html id of the source element
 * @param {string} targetId Html id of the target element
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addOpenPageTransition(
    sourcePage, targetPage, sourceId, targetId, controls = null) {
  // setup
  let source = $(`#${sourceId}`), target = $(`#${targetId}`),
      eventType = 'click';
  if (sourceId.startsWith('group')) {
    source = source.find(`[id^=actionable_]`);
  }
  if (targetId.startsWith('group')) {
    target = target.closest(`[id^=page]`);
  }
  if (source.length == 0) return;
  let tagName = source.prop('tagName'), inputType = source.attr('type');
  switch (tagName) {
    case 'INPUT':
      switch (inputType) {
        case 'text':
        case 'password':
          eventType = 'keypress';
          break;
      }
      break;
  }

  // init
  target.hide();  // we expect target page to be closed anyway but just in case

  // event
  if (controls && controls.hasOwnProperty('eventType'))
    eventType = controls.eventType;
  if (eventType == 'keyboard') eventType = 'keypress';
  const sourcePageNumber = getPageNumber(sourceId);
  $(`#page${sourcePageNumber}`).addClass('connected');
  if (controls && controls.hasOwnProperty('preconditionVisited')) {
    for (let i = 0; i < controls['preconditionVisited'].length; i++) {
      if (!source.attr('class').includes(
              `gminiwob-precondition-${controls['preconditionVisited'][i]}`))
        source.addClass(
            `gminiwob-precondition-${controls['preconditionVisited'][i]}`);
    }
  }
  source.on(eventType, function() {
    if (controls && controls.hasOwnProperty('preconditionVisited')) {
      for (let i = 0; i < controls['preconditionVisited'].length; i++) {
        if (!$(`#${controls['preconditionVisited'][i]}`)
                 .attr('class')
                 .includes('visited'))
          return;
      }
    }
    $(`#page${sourcePageNumber}`).hide();
    return target.show();
  });
}

/**
 * Creates and adds a submit transition.
 * The transitions extracts the page of the sourceId element and adds an
 * annotation to the page that says it is a submission page. Then, it adds
 * an event to the sourceId element such that when fired, the episode will
 * terminate and website will be submitted.
 * If conceptual is given via controls, reward is abstract reward where
 * only visiting the correct element is relevant not the profile.
 * Only sourceId parameter is used and others are added to adhere to the same
 * standard as other transition functions.
 * controls=(conceptual,preconditionVisited)
 * @param {number} sourcePage Index of the source page
 * @param {number} targetPage Index of the target page
 * @param {string} sourceId Html id of the source element
 * @param {string} targetId Html id of the target element
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addSubmitTransition(
    sourcePage, targetPage, sourceId, targetId, controls = null) {
  // setup
  let source = $(`#${sourceId}`);
  if (sourceId.startsWith('group')) {
    source = source.find(`[id^=actionable_]`);
  }
  if (source.length == 0) return;

  // init

  // event
  let sourcePageNumber = getPageNumber(sourceId);
  $(`#page${sourcePageNumber}`).addClass('submit-page');
  if (controls && controls.hasOwnProperty('preconditionVisited')) {
    for (let i = 0; i < controls['preconditionVisited'].length; i++) {
      if (!source.attr('class').includes(
              `gminiwob-precondition-${controls['preconditionVisited'][i]}`))
        source.addClass(
            `gminiwob-precondition-${controls['preconditionVisited'][i]}`);
    }
  }
  console.log(['controls for submit', controls]);
  source.on('click', function() {
    if (!source.attr('class').includes('visited')) source.addClass('visited');
    if (controls && controls.hasOwnProperty('preconditionVisited')) {
      for (let i = 0; i < controls['preconditionVisited'].length; i++) {
        if (!$(`#${controls['preconditionVisited'][i]}`)
                 .attr('class')
                 .includes('visited'))
          return;
      }
    }
    let profile = JSON.parse($('#query').html());
    let submittedFields = {};
    for (let i = 0; i < CONCEPTS.length; i++) {
      let inputElement = $('#' + getActionableIdForConcept(CONCEPTS[i]));
      if (inputElement.length == 0) {
        inputElement = $(`[id^=${getActionableIdForConcept(CONCEPTS[i])}]`)
                           .filter(function() {
                             return $(this).prop('checked');
                           });
      }
      if (inputElement.attr('type') == 'checkbox') {
        submittedFields[CONCEPTS[i]] = inputElement.prop('checked');
      } else {
        submittedFields[CONCEPTS[i]] = inputElement.val();
      }
    }
    profile_len = 0.0;
    jQuery.each(profile, function(field) {
      profile_len += 1;
    });
    if (profile.hasOwnProperty('_dummy')) profile_len -= 1;
    let r = computeFormReward(submittedFields, profile);
    if (controls && controls.hasOwnProperty('conceptual') &&
        controls.conceptual) {
      r = computeConceptualReward();
    }
    r = r === 1.0           ? controls.hasOwnProperty('taskSuccessScale') ?
                              r * controls.taskSuccessScale :
                              r :
        profile_len === 0.0 ? controls.hasOwnProperty('taskSuccessScale') ?
                              controls.taskSuccessScale :
                              1.0 :
                              -1.0;
    core.endEpisode(r, r > 0);
  });
}

/**
********************************************************************************
* Function that are visible in html environments.
********************************************************************************
**/
/**
 * Creates and returns a profile from available concepts.
 * Concepts should be core concepts where a profile field is needed and they
 * are important for reward estimation. Otherwise, they should be excluded
 * from the list of concepts when invoking this function.
 * If singleValue is true, then a single deterministic value for each field
 * will be used. Otherwise, the value is sampled from a database.
 * If isBinaryProfile is true, the default profile fields are binary fields
 * and they will have true/false values.
 * @param {!Array<string>} concepts List of concepts to iterate over
 * @param {boolean=} singleValue Should profiles have a single dummy value or a
 *     database
 * @param {boolean=} isBinaryProfile Is binary profiles the default
 * @param {!ObjType=} mainProfile [Ignored] Main profile to copy from.
 * @return {!ObjType} A dictionary of profile as a set of key and value pairs
 */
function createProfileFromConcepts(
    concepts, singleValue = false, isBinaryProfile = false, mainProfile = null) {
  let profile = {};
  for (let i = 0; i < concepts.length; i++) {
    if (concepts[i] == passwordConcept) {
      profile[concepts[i]] =
          singleValue ? 'pass' : ui_utils.generateString(2, 6);
    } else if (concepts[i] == captchaConcept) {
      profile[concepts[i]] =
          singleValue ? 'CaPtChA' : ui_utils.generateString(2, 6);
    } else if (concepts[i] == firstnameConcept) {
      profile[concepts[i]] = singleValue ?
          'John' :
          core.sample(ui_utils.FIFTY_NAMES).toLowerCase();
    } else if (concepts[i] == lastnameConcept) {
      profile[concepts[i]] =
          singleValue ? 'Wick' : core.sample(ui_utils.LAST_NAMES).toLowerCase();
    } else if (concepts[i] == usernameConcept) {
      profile[concepts[i]] = singleValue ?
          'john@email.com' :
          core.sample(ui_utils.FIFTY_NAMES).toLowerCase().concat('@email.com');
    } else if (concepts[i] == addressline1Concept) {
      profile[concepts[i]] = singleValue ?
          '413 E Mountain View' :
          alert('Add address line 1 database!');
    } else if (concepts[i] == addressline2Concept) {
      profile[concepts[i]] =
          singleValue ? '2042' : alert('Add address line 1 database!');
    } else if (concepts[i] == cityConcept) {
      profile[concepts[i]] =
          singleValue ? 'Mountain View' : alert('Add city database!');
    } else if (concepts[i] == zipcodeConcept) {
      profile[concepts[i]] =
          singleValue ? '95123' : alert('Add zip code database!');
    } else if (concepts[i] == stateConcept) {
      profile[concepts[i]] = singleValue ? core.sample(['CA', 'NY']) :
                                           alert('Add state database!');
    } else if (concepts[i] == fullnameConcept) {
      profile[concepts[i]] =
          singleValue ? 'John Wick' : alert('Add fullname database!');
    } else if (concepts[i] == ccnumberConcept) {
      profile[concepts[i]] = singleValue ?
          '0000 0000 0000 0000' :
          alert('Add credit card number database!');
    } else if (concepts[i] == cctypeConcept) {
      profile[concepts[i]] = singleValue ?
          core.sample(['Credit Card', 'Debit Card']) :
          alert('Add credit card number database!');
    } else if (concepts[i] == ccexpirationdateConcept) {
      profile[concepts[i]] = singleValue ?
          '2025' :
          alert('Add credit card expiration date database!');
    } else if (concepts[i] == cccvvConcept) {
      profile[concepts[i]] =
          singleValue ? '123' : alert('Add credit card CVV database!');
    } else if (concepts[i] == departureairportConcept) {
      profile[concepts[i]] = singleValue ?
          'San Francisco Airport (SFO)' :
          alert('Add departureairport database!');
    } else if (concepts[i] == destinationairportConcept) {
      profile[concepts[i]] = singleValue ?
          'Los Angeles Airport (LAX)' :
          alert('Add destinationairport database!');
    } else if (concepts[i] == departuredateConcept) {
      profile[concepts[i]] =
          singleValue ? '10/02/2020' : alert('Add departuredate database!');
    } else if (concepts[i] == destinationdateConcept) {
      profile[concepts[i]] =
          singleValue ? '10/03/2020' : alert('Add destinationdate database!');
    } else if (concepts[i] == cabinConcept) {
      profile[concepts[i]] = singleValue ? core.sample(['Economy', 'First']) :
                                           alert('Add cabinConcept database!');
    } else if (concepts[i] == flighttypeConcept) {
      profile[concepts[i]] = singleValue ?
          core.sample(['Oneway', 'Roundtrip']) :
          alert('Add flighttypeConcept database!');
    } else if (concepts[i] == numberofpeopleConcept) {
      profile[concepts[i]] = singleValue ?
          core.sample(['1', '2', '3', '4', '5']) :
          alert('Add numberofpeopleConcept database!');
    } else if (getBinaryConcepts().includes(concepts[i]) || isBinaryProfile) {
      profile[concepts[i]] = core.sample([true, false]);
    }
  }
  return [profile, mainProfile];
}

/**
 *Functions that are visible for environment design.
 **/
/**
 * Compare form inputs to ground truth fields and conpute task success.
 * Comparison is done directly, corresponding fields should exactly match.
 * For example, CA and california are two different values even though
 * they might correspond to the same thing.
 * @param {!Array<string>} submittedFields Fields collected from the website
 * @param {!Array<string>} goldFields Ground truth fields
 * @return {number} Task success rate
 */
function computeFormReward(submittedFields, goldFields) {
  let reward = 0.0;
  let size = 0.0;
  console.log([submittedFields, goldFields]);
  jQuery.each(submittedFields, function(key, value) {
    reward += goldFields[key] === value ? 1.0 : 0.0;
  });
  jQuery.each(goldFields, function(field) {
    size += 1;
  });
  if (goldFields.hasOwnProperty('_dummy')) size -= 1;
  console.log([reward, size, reward / size]);
  if (size == 0) return 0.0;
  return reward / size;
}

/**
 * Computes task success rate for abstract navigation where only visiting
 * correct elements is important but not profiles. If elements are visited,
 * they will be annotated as 'visited'. If they are visited and they appear
 * in the list of concepts, then it is a success.
 * @return {number} Task success rate
 */
function computeConceptualReward() {
  let reward = 0.0;
  let size = 0.0;
  let goldFields = JSON.parse($('#query').html());
  console.log(CONCEPTS);
  $('*[id^="actionable"]').each(function() {
    if ($(this).hasClass('visited') &&
        CONCEPTS.includes($(this).attr('id').substring('actionable_'.length)))
      reward += 1.0;
  });
  jQuery.each(goldFields, function(field) {
    size += 1;
  });
  if (goldFields.hasOwnProperty('_dummy')) size -= 1;
  console.log([reward, size]);
  if (size == 0) return 0.0;
  return reward / size;
}

/**
 * A wrapper around _appendEmptyPagesWithForms with the root is prespecified.
 * @param {number} numPages Total number of empty pages
 * @return {!ObjType} A jquery object where empty pages are added
 */
function appendEmptyPagesWithForms(numPages) {
  return _appendEmptyPagesWithForms($('#area'), numPages, 'core');
}

/**
 * Create and add a conceptual primitive to the given page.
 * Depending on the concept, the primitive will either be a form related or
 * global primitive. A form related primitive will be added to a specific
 * root while otherwise it will be insterted before the same specific root.
 * @param {number} sourcePage Index of the source page
 * @param {string} concept Name of the concept
 * @param {!ObjType} profile User profile object
 * @param {boolean} isCore If the primitive is core or not
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addGroup(sourcePage, concept, profile, isCore, controls = null) {
  let formGroup = false;
  switch (concept) {
    case inputgroupConcept:  // useful for passive groups
    case firstnameConcept:
    case lastnameConcept:
    case passwordConcept:
    case usernameConcept:
    case stayloggedinConcept:
    case remembermeConcept:
    case captchaConcept:
    case addressline1Concept:
    case addressline2Concept:
    case cityConcept:
    case zipcodeConcept:
    case stateConcept:
    case fullnameConcept:
    case ccexpirationdateConcept:
    case ccnumberConcept:
    case cccvvConcept:
    case departureairportConcept:
    case destinationairportConcept:
    case departuredateConcept:
    case destinationdateConcept:
    case numberofpeopleConcept:
      formGroup = true;
      _addInputGroup(sourcePage, concept, isCore, controls);
      break;
    case navbarConcept:
      _addNavbar(sourcePage, concept, controls);
      break;
    case footerConcept:
      _addFooter(sourcePage, concept, controls);
      break;
    case forgotusernameConcept:
    case forgotpasswordConcept:
      formGroup = true;
      addLinkGroup(sourcePage, concept, controls);
      break;
    case carouselConcept:
      _addCarousel(sourcePage, concept, controls);
      break;
    case deckConcept:
      _addDeck(sourcePage, concept, controls);
      break;
    case headerConcept:
      _addHeader(sourcePage, concept, controls);
      break;
    case nextConcept:
    case submitConcept:
      formGroup = true;
      _addButton(sourcePage, concept, controls);
      break;
    case cartConcept:
      addCart(sourcePage, concept, controls);
      break;
    case dealmediaConcept:
      addMedia(sourcePage, concept, controls);
      break;
    case singleselectionConcept:  // useful for passive groups
    case cctypeConcept:
    case flighttypeConcept:
    case cabinConcept:
      formGroup = true;
      _addSingleSelectionButtonGroup(sourcePage, concept, controls);
      break;
  }
  let parent = $(`#page${sourcePage}`);
  let sampledConcept = sampleConcept(concept, sourcePage);
  console.log([concept, sourcePage]);
  if (formGroup) {  // form related primitives are added to #mainform root
    parent = parent.find('#mainform');
    console.log(REGISTERED_CONCEPTS);
    parent.append(sampledConcept);
  } else {  // other primitives are inserted before #mainform
    sampledConcept.insertBefore(
        parent.find('#mainform').parentsUntil(`#page${sourcePage}`).last());
  }
}

/**
 * Add a new transition.
 * Transitions are used to enable element level dependencies and page
 * transitions and submissions. An example element dependency would be
 * showing an otherwise invisible 'password' element after 'username' element
 * is interacted with.
 * @param {string} type Type of the transition
 * @param {number} sourcePage Index of the source page
 * @param {number} targetPage Index of the target page
 * @param {string} sourceGroup Html id of the source element
 * @param {string} targetGroup Html id of the target element
 * @param {string} concept Name of the concept
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function addTransition(
    type, sourcePage, targetPage, sourceGroup, targetGroup, concept, controls) {
  switch (type) {
    case 'addShowHideTransition':
      addShowHideTransition(
          sourcePage, targetPage, sourceGroup, targetGroup, controls);
      break;
    case 'addOpenPageTransition':
      addOpenPageTransition(
          sourcePage, targetPage, sourceGroup, targetGroup, controls);
      break;
    case 'addSubmitTransition':
      addSubmitTransition(
          sourcePage, targetPage, sourceGroup, targetGroup, controls);
  }
}

/**
 * Creates a callable that adds nodes and profile to the website.
 * It creates a callable such that the callable generates a global list of all
 * primitives and also a user profile. It adds each symbolic action to the
 * website. The callable is global in which it can be run anywhere to generate
 * the website.
 * @param {!Array<!ObjType>} symbolicActions A list of symbolic action objects
 * @param {boolean=} addDummyField If true, add a dummy profile so that profile
 *     is always non-empty
 * @param {boolean=} isBinaryProfile Is binary profile values the default
 *
 * @return {!ObjType} Return a callable that generates pages
 */
function addNodes(
    symbolicActions, addDummyField = false, isBinaryProfile = false) {
  NUM_PAGES = symbolicActions.num_pages;  // get pages first
  nodeCreateFn = function() {
    $('#area').empty();
    console.log($(`#area`).html());
    let profile = {};
    CONCEPTS = [];
    REGISTERED_CONCEPTS = [];
    appendEmptyPagesWithForms(NUM_PAGES);
    let mainProfile = null;
    let sampledProfile = null;
    for (let i = 0; i < symbolicActions.actions.length; i++) {
      const action = symbolicActions.actions[i];
      if (!action.is_transition) {
        if (action.is_core_concept) {
          // Add concept for reward computation
          CONCEPTS.push(action.concept);
          // Create a profile from new concept
          [sampledProfile, mainProfile] = createProfileFromConcepts(
                  [action.concept], true, isBinaryProfile, mainProfile);
          $.extend(
              profile,
              sampledProfile);

        }  // this concept is required for reward and profile
        sourcePage = action.source_page;
        if (sourcePage == -1) sourcePage = [...Array(NUM_PAGES).keys()];
        if (!Array.isArray(sourcePage)) sourcePage = [sourcePage];
        for (let j = 0; j < sourcePage.length; j++) {
          addGroup(
              sourcePage[j], action.concept, profile, action.is_core_concept,
              action.controls);
        }
      }
    }
    // Creates a profile.
    if (addDummyField) {
      profile['_dummy'] = null;
    }
    $('#query').html(JSON.stringify(profile));
  };
  nodeCreateFn();
  for (let i = 0; i < NUM_PAGES; i++) {
    $(`#page${i}`).show();
    $(`#page${i}`).find('*').show();
  }
  const siteMap = core.getDOMInfo();
  $('#area').empty();
  return siteMap;
}

/**
 * Creates a callable that adds edges to the website.
 * It creates a callable such that the callable generates a list of events
 * which builds transitions between primitives and web pages. It returns true
 * if runs successfully. The callable is global in which it can be run
 * anywhere to generate the website.
 * @param {!Array<!ObjType>} symbolicActions A list of symbolic action objects
 * @param {boolean=} isBinaryProfile Is binary profile values the default
 *
 * @return {!ObjType} Return a callable that generates transitions
 */
function addEdges(symbolicActions, isBinaryProfile = false) {
  console.log(symbolicActions);
  edgeCreateFn = function() {
    for (let i = 0; i < symbolicActions.actions.length; i++) {
      const action = symbolicActions.actions[i];
      if (action.is_transition) {
        if (USE_CONCEPTUAL) {
          action.controls['conceptual'] = true;  // set conceptual here
        }
        console.log('adding edge', action);
        sourceGroup = action.source_group;
        if (!Array.isArray(sourceGroup)) sourceGroup = [sourceGroup];
        for (let j = 0; j < sourceGroup.length; j++) {
          if (action.type == 'addShowHideTransition' ||
              !action.hasOwnProperty('target_page') ||
              action.target_page < NUM_PAGES) {
            addTransition(
                action.type, action.source_page, action.target_page,
                sourceGroup[j], action.target_group, action.concept,
                action.controls);
          } else if (
              action.controls.hasOwnProperty('shouldSubmitOnFinalPage') &&
              action.controls.shouldSubmitOnFinalPage) {
            addTransition(
                'addSubmitTransition', action.source_page, action.target_page,
                sourceGroup[j], action.target_group, action.concept,
                action.controls);
          }
        }
      }
    }
  };
  return true;
}

/**
 * Connect the web page graph if it is disconnected.
 * This will first create necessary elements with next or submit concepts if
 * they are missing. Using these elements, it will add required transitions.
 * Submit will only be added to the last page and next to non-last pages.
 * @param {!Array<!ObjType|!Array<string>>=} controls Optional controls
 */
function connectGraph(controls) {
  for (let pageIndex = 1; pageIndex <= NUM_PAGES; pageIndex++) {
    const page = $(`#page${pageIndex - 1}`);
    if (!page.attr('class').includes('connected')) {
      page.addClass('connected');
      let concept = nextConcept;
      if (pageIndex == NUM_PAGES) concept = submitConcept;
      console.log($(`#${getActionableIdForConcept(concept, pageIndex - 1)}`).length);
      if ($(`#${getActionableIdForConcept(concept, pageIndex - 1)}`).length ==
          0)
        _addButton(
            pageIndex - 1, concept,
            controls);  // add next button to previous page
      const sampledConcept = sampleConcept(concept, pageIndex - 1);
      parent = page.find('#mainform');
      parent.append(sampledConcept);
      if (pageIndex == NUM_PAGES &&
          !$(`#page${pageIndex - 1}`).hasClass('submit-page')) {
        console.log('Connecting via submit transition:');
        console.log($(`#${getActionableIdForConcept(concept, pageIndex - 1)}`)
                        .attr('id'));
        addSubmitTransition(
            pageIndex - 1, pageIndex - 1,
            getActionableIdForConcept(concept, pageIndex - 1),
            `page${pageIndex - 1}`, controls);
      } else if (pageIndex < NUM_PAGES) {
        console.log('Connecting via open page transition:');
        console.log($(`#${getActionableIdForConcept(concept, pageIndex - 1)}`)
                        .attr('id'));
        addOpenPageTransition(
            pageIndex - 1, pageIndex,
            getActionableIdForConcept(concept, pageIndex - 1),
            `page${pageIndex}`, controls);
      }
    }
  }
}

/**
 * Extract related fields from the website and estimate reward as potential.
 * Reward estimation is task success -- percentage of fields that are correct.
 * Potential is equal to this but it can be estimated at every step to compute
 * the potential difference as the stepwise reward.
 * @param {boolean=} checkVisited If true, check if an element is visited to
 *     use in potential estimation
 *
 * @return {number} A potential -- percentage of elements that have correct
 *     value given the user profile
 */
function potential(checkVisited = false) {
  const submittedFields = {};
  for (let i = 0; i < CONCEPTS.length; i++) {
    let inputElement = $('#' + getActionableIdForConcept(CONCEPTS[i]));
    if (inputElement.length == 0) {
      inputElement = $(`[id^=${getActionableIdForConcept(CONCEPTS[i])}]`)
                         .filter(function() {
                           return $(this).prop('checked');
                         });
    }
    if (!checkVisited || inputElement.hasClass('visited')) {
      if (inputElement.attr('type') == 'checkbox') {
        submittedFields[CONCEPTS[i]] = inputElement.prop('checked');
      } else {
        submittedFields[CONCEPTS[i]] = inputElement.val();
      }
    } else {
      submittedFields[CONCEPTS[i]] = null;
    }
  }
  return computeFormReward(submittedFields, JSON.parse($('#query').html()));
}

/**
 * A wrapper to compute conceptual potential.
 * The difference is that this assumes the website is evaluated only
 * conceptually -- profile is not relevant.
 * @return {number} A potential -- percentage of elements that have correct
 *     value given the user profile in conceptual setting
 */
function conceptualPotential() {
  return computeConceptualReward();
}

/**
 * Visit a conceptual group using the HTML id of the corresponding element and a
 * text value to be used to interact with the element. This function is an
 * alternative to MiniWoB/Selenium interface where action type click/keyboard is
 * required from outside. This function handles that implicitly by not taking
 * actual click/keyboard actions but updating 'value' attribute and then firing
 * corresponding events. Note that although this handles a lot of cases, this
 * will be limiting with 3rd party libraries.
 * @param {string} id Id of an element to visit and take action
 * @param {string=} value A text value to use to act on the element
 * @param {number=} timeout Delay for running the action
 */
function visitGroup(id, value = null, timeout = 0) {
  setTimeout(function() {
    const pageIndex = getPageNumber(id, true);
    let concept = id.substring(id.indexOf('_') + 1);
    concept = concept.substring(0, concept.indexOf('_'));
    if ((pageIndex > 0 &&
         $(`#page${pageIndex}`).find(`#${id}`).is(':visible') &&
         $(`#page${pageIndex}`)
                 .find(`#${id}`)
                 .parents('.disableElements')
                 .length == 0) ||
        ($(`#${id}`).is(':visible') &&
         $(`#${id}`).parents('.disableElements').length == 0)) {
      // var group = $(`#page${pageIndex}`).find(`#${id}`);
      // var element =
      // group.find(`[id^=${getActionableIdForConcept(concept)}]`);
      let element = $(`#${id}`);
      element.addClass('visited');
      if ((element.is('input') &&
           (element.attr('type') == 'text' ||
            element.attr('type') == 'number' ||
            element.attr('type') == 'email' ||
            element.attr('type') == 'password')) ||
          element.is('select')) {
        element.trigger('keypress');
        element.val(value != null ? value : '');
      } else {
        if (element.is('input') && element.attr('type') == 'radio') {
          const root = element.closest(`.${concept}root`);
          root.find(`[id^=actionable_]`)
                        .filter(function() {
                          return $(this).attr('value') == value;
                        })
                        .click();
          root.find(`[id^=actionable_]`).addClass('visited');
        } else if (
            value == true ||
            !(element.is('input') && (element.attr('type') == 'checkbox'))) {
          element.click();
        }
      }
    }
  }, timeout);
}
