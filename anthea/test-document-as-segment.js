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

/**
 * To run this test, visit the URL:
 *     http://.../anthea.html?test=test-document-as-segment.js
 * and then click around through the rating process.
 */
const testSrcDoc =
    'Introduction\\n\\n\u200b\u200b' +
    'Google is not a conventional company\u200b. \u200b\u200b' +
    'We do not intend to become one\u200b. \u200b\u200b' +
    'Throughout Google\u200b’s evolution as a privately held ' +
    'company\u200b, we have managed Google differently\u200b. \u200b\u200b' +
    'We have also emphasized an atmosphere of creativity and ' +
    'challenge\u200b, which has helped us provide unbiased\u200b, ' +
    'accurate and free access to information for those who rely on us ' +
    'around the world\u200b.\\n\\n\u200b\u200bNow the time has come for the ' +
    'company to move to public ownership\u200b. \u200b\u200bThis change ' +
    'will bring important benefits for our employees\u200b, for our present ' +
    'and future shareholders\u200b, for our customers\u200b, and most of all ' +
    'for Google users\u200b. \u200b\u200bBut the standard structure of ' +
    'public ownership may jeopardize the independence and focused ' +
    'objectivity that have been most important in Google\u200b’s past ' +
    'success and that we consider most fundamental for its future\u200b. ' +
    '\u200b\u200bTherefore\u200b, we have implemented a corporate ' +
    'structure that is designed to protect Google\u200b’s ability to ' +
    'innovate and retain its most distinctive characteristics\u200b. ' +
    '\u200b\u200bWe are confident that\u200b, in the long run\u200b, this ' +
    'will benefit Google and its shareholders\u200b, old and new\u200b. ' +
    '\u200b\u200bWe want to clearly explain our plans and the reasoning ' +
    'and values behind them\u200b. \u200b\u200bWe are delighted you are ' +
    'considering an investment in Google and are reading this ' +
    'letter\u200b.\\n\\n\u200b\u200bSergey and I intend to write you a ' +
    'letter like this one every year in our annual report\u200b. \u200b\u200b' +
    'We\u200b’ll take turns writing the letter so you\u200b’ll hear directly ' +
    'from each of us\u200b. \u200b\u200bWe ask that you read this letter ' +
    'in conjunction with the rest of this prospectus\u200b.\\n\\n\u200b\u200b' +
    'Serving end users\\n\\n\u200b\u200b' +
    'Sergey and I founded Google because we believed we could ' +
    'provide an important service to the world\u200b-\u200binstantly ' +
    'delivering relevant information on virtually any topic\u200b. ' +
    '\u200b\u200bServing our end users is at the heart of what we do and ' +
    'remains our number one priority\u200b.';

const testTgtDoc =
    'Einführung\\n\u200b\u200b' +
    'Google ist kein herkömmliches Unternehmen\u200b. \u200b\u200bWir ' +
    'haben nicht die Absicht\u200b, eins zu werden\u200b. \u200b\u200bIm ' +
    'Laufe der Entwicklung von Google als privat geführtes Unternehmen ' +
    'haben wir Google unterschiedlich verwaltet\u200b. \u200b\u200bWir ' +
    'haben auch Wert auf eine Atmosphäre der Kreativität und Herausforderung ' +
    'gelegt\u200b, die uns dabei geholfen hat\u200b, denjenigen\u200b, die ' +
    'sich auf uns auf der ganzen Welt verlassen\u200b, einen ' +
    'unvoreingenommenen\u200b, genauen und kostenlosen Zugang zu ' +
    'Informationen zu ermöglichen\u200b.\\n\\n\u200b\u200b' +
    'Nun ist es an der Zeit\u200b, dass das Unternehmen in öffentliches ' +
    'Eigentum übergeht\u200b. \u200b\u200bDiese Änderung wird unseren ' +
    'Mitarbeitern\u200b, unseren derzeitigen und zukünftigen ' +
    'Aktionären\u200b, unseren Kunden und vor allem den ' +
    'Google\u200b-\u200bNutzern wichtige Vorteile bringen\u200b. ' +
    '\u200b\u200bDie Standardstruktur des öffentlichen Eigentums kann ' +
    'jedoch die Unabhängigkeit und gezielte Objektivität gefährden\u200b, ' +
    'die für den bisherigen Erfolg von Google am wichtigsten waren und die ' +
    'wir für die Zukunft des Unternehmens als grundlegend erachten\u200b. ' +
    '\u200b\u200bAus diesem Grund haben wir eine Unternehmensstruktur ' +
    'implementiert\u200b, die darauf abzielt\u200b, die Innovationsfähigkeit ' +
    'von Google zu schützen und seine markantesten Merkmale ' +
    'beizubehalten\u200b. \u200b\u200bWir sind zuversichtlich\u200b, dass ' +
    'dies auf lange Sicht Google und seinen alten und neuen Aktionären ' +
    'zugute kommen wird\u200b. \u200b\u200bWir möchten unsere Pläne sowie ' +
    'die dahinter stehenden Überlegungen und Werte klar erläutern\u200b. ' +
    '\u200b\u200bWir freuen uns\u200b, dass Sie über eine Investition in ' +
    'Google nachdenken und diesen Brief lesen\u200b.\\n\\n\u200b\u200b' +
    'Sergey und ich beabsichtigen\u200b, Ihnen jedes Jahr in unserem ' +
    'Jahresbericht einen Brief wie diesen zu schreiben\u200b. ' +
    '\u200b\u200bWir werden den Brief abwechselnd schreiben\u200b, sodass ' +
    'Sie direkt von jedem von uns hören können\u200b. \u200b\u200bWir ' +
    'bitten Sie\u200b, diesen Brief zusammen mit dem Rest dieses Prospekts ' +
    'zu lesen\u200b.\\n\u200b\u200b' +
    'Betreuung von Endbenutzern\\n\u200b\u200b' +
    'Sergey und ich haben Google gegründet\u200b, weil wir glaubten\u200b, ' +
    'wir könnten der Welt einen wichtigen Dienst bieten – die sofortige ' +
    'Bereitstellung relevanter Informationen zu praktisch jedem Thema\u200b. ' +
    '\u200b\u200bDie Betreuung unserer Endverbraucher steht im Mittelpunkt ' +
    'unseres Handelns und bleibt unsere oberste Priorität\u200b.';

const testProjectTSVData = `{"source_language":"en","target_language":"en","paralet_sentences":4,"paralet_tokens":150}
${testSrcDoc}\t${testTgtDoc}\tdoc-founders-letter-2004\tsystem-GL
      `;
const testProjectName = 'Google-MQM-Test-Document-As-Segment';
const testTemplateName = 'MQM';
const activeName = antheaManager.activeName(testProjectName, testTemplateName);
try {
  const activeDataKey = antheaManager.ACTIVE_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeDataKey);
  const activeResultsKey =
      antheaManager.ACTIVE_RESULTS_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeResultsKey);
} catch (err) {
  console.log('Caught exception (harmless if for "no such item": ' + err);
}
antheaManager.createActive(
    testTemplateName, testProjectName, testProjectTSVData,
    20 /* HOTW rate */);
