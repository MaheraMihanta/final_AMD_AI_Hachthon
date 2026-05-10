# pill-j8vgy-o5udx-7xdez-ehbb > 2025-03-13 11:36pm
https://universe.roboflow.com/rf-100-vl/pill-j8vgy-o5udx-7xdez-ehbb

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Light Front](#light-front)
  - [No Pill Back](#nopill-back)
  - [No Pill Front](#nopill-front)
  - [Pill Back](#pill-back)
  - [Pill Front](#pill-front)

# Introduction
This dataset is designed for the detection and classification of pill packages from different perspectives. It includes five classes: Light Front, No Pill Back, No Pill Fornt, Pill Back, and Pill Front. Each class is defined based on the orientation and contents of the pill package.

# Object Classes

## Light Front
### Description
Light Front refers to pill packages facing forward with visible pills that appear to be lit from behind, creating a translucent effect.

### Instructions
Annotate the entire pill package, ensuring the label covers the full extent of the package’s translucent appearance. Do not include reflections or shadows apart from the package itself.

## No Pill Back
### Description
No Pill Back represents pill packages viewed from the back, where the pills are not visible, and the package often has text or patterns. The seal has been broken for all pill compartments.

### Instructions
Annotate the entire package area, focusing on the back pattern or text. Ensure to exclude any reflection or partial obstructions outside of the primary package surface.

## No Pill Front
### Description
No Pill Front includes pill packages facing forward without visible pills, either due to empty blister pockets or opaque coverage.

### Instructions
Outline the entire visible front of the package. Focus on areas where the pill shape is indiscernible and ensure not to include areas beyond the defined package lines.

## Pill Back
### Description
Pill Back consists of pill packages viewed from the back, where pills create raised shapes that are visible through their packaging. The aluminum film is not broken.

### Instructions
Draw bounding boxes around each raised section corresponding to individual pills. Do not include flat areas of the package that are void of pill contours.

## Pill Front
### Description
Pill Front identifies pill packages from the front where pills distinctly protrude into visible, raised blisters. Pill can be of diverse colors. 

### Instructions
Label each visible blister encapsulating a pill, focusing on the raised portions. Exclude areas of the package that do not contain visible blisters or pills.