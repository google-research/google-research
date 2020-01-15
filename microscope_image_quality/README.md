Focus_check script generates whole-plate montages per channel for inspecting
data quality.

PUBLICATION:
Yang, S. J., Berndl, M., Ando, D. M., Barch, M., Narayanaswamy, A. ,
Christiansen, E., Hoyer, S., Roat, C., Hung, J., Rueden, C. T., Shankar, A.,
Finkbeiner, S., & and Nelson, P. (2018), "Assessing microscope image focus
quality with deep learning", BMC BioInformatics 19(1).

Link to publication: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2087-4

INSTALLING MACRO:
https://github.com/fiji/microscope-image-quality


Fiji macros are scripts that increase throughput of image analysis, similar to
that of already existing plugins. There are two ways of creating a macro:
opening and editing an existing macro or recording your steps as you go through
a protocol.

OPEN AND EDIT EXISTING MACRO:
1. Go to PLUGINS > MACROS > EDIT...
2. Find desired protocol and open
3. Edit desired protocol and save

RECORD STEPS:
1. Go to PLUGINS > MACROS > RECORD...
2. Make sure at the top of the recorder pop-up window:
  1. "Record: Macro"
  2. "Name:" is set to the desired name of your macro (ex.
      "Focus_Quality_Check"). You do not need to include the extension as you
      can add that later when saving your macro
3. Start going through your protocol, step-by-step
4. Hit "Create" at the top of the recorder pop-up window
5. Save your protocol

Once you are done creating your protocol, hit "Run" at the bottom of the editor
window.

TO DO:
(comment by samuely) Have one master script that we add flags for turning on/off
certain features, such as the "single site" analysis or "with percentage". Since
a lot of code is shared, it's easier to keep track of a single file.
Documentation about the various flag options can go in the code rather than in
this doc.
