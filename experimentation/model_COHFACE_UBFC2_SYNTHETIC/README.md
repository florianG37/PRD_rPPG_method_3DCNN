# COHFACE & UBFC2 & SYNTHETIC datasets Training

<em> Author : Florian GIGOT </em>

## Overview

### Number of session : 14 (save-1)

- 1 session  = 10 epochs
- COHFACE = compressed video
- UBFC2 = not compressed video

#### Validation tests
<table>
    <thead>
        <th>Dataset</th>
        <th>Result (accuracy)</th>
    </thead>
    <tbody>
        <tr>
            <td>SYNTHETIC</td>
            <td>53.95%</td>
        </tr>
        <tr>
            <td>COHFACE</td>
            <td>34,97%</td>
        </tr>
        <tr>
            <td>UBFC2</td>
            <td>11.92%</td>
        </tr>
    </tbody>
</table>

#### Real-world tests

<table>
    <thead>
        <th>Video</th>
        <th>Mean error </th>
    </thead>
    <tbody>
        <tr>
            <td>COHFACE Subject 1 vid 0</td>
            <td>6.38 bpm</td>
        </tr>
        <tr>
            <td>COHFACE Subject 2 vid 0</td>
            <td>16.04 bpm</td>
        </tr>
        <tr>
            <td>UBFC2 Suject 1</td>
            <td>19.71 bpm</td>
        </tr>
        <tr>
            <td>UBFC2 Subject 3</td>
            <td>15.36 bpm</td>
        </tr>
    </tbody>
</table>

### Number of session : 20 (save-2)

- 1 session  = 10 epochs
- COHFACE = compressed video
- UBFC2 = not compressed video

#### Validation tests
<table>
    <thead>
        <th>Dataset</th>
        <th>Result (accuracy)</th>
    </thead>
    <tbody>
        <tr>
            <td>SYNTHETIC</td>
            <td>74,84%</td>
        </tr>
        <tr>
            <td>COHFACE</td>
            <td>31.90%</td>
        </tr>
        <tr>
            <td>UBFC2</td>
            <td>17.22%</td>
        </tr>
    </tbody>
</table>


#### Real-world tests

<table>
    <thead>
        <th>Video</th>
        <th>Mean error </th>
    </thead>
    <tbody>
        <tr>
            <td>COHFACE Subject 1 vid 0</td>
            <td>5.76 bpm</td>
        </tr>
        <tr>
            <td>COHFACE Subject 2 vid 0</td>
            <td>15.55 bpm</td>
        </tr>
        <tr>
            <td>UBFC2 Suject 1</td>
            <td>19.71 bpm</td>
        </tr>
        <tr>
            <td>UBFC2 Subject 3</td>
            <td>15.36 bpm</td>
        </tr>
    </tbody>
</table>


## Details

### 14 sessions

#### Tests validation
![Tests validation](./imgs/test_Validation-save1.JPG)
#### COHFACE Subject 1 vid 0
![COHFACE Subject 1 vid 0](./imgs/COHFACE-1-0-save1.JPG)
#### COHFACE Subject 2 vid 0
![COHFACE Subject 2 vid 0](./imgs/COHFACE-2-0-save1.JPG)
#### UBFC2 Suject 1
![UBFC2 Suject 1](./imgs/UBFC-1-save1.JPG)
#### UBFC2 Subject 3
![UBFC2 Subject 3](./imgs/UBFC-3-save1.JPG)

### 20 sessions

#### Tests validation
![Tests validation](./imgs/test_Validation-save2.JPG)
#### COHFACE Subject 1 vid 0
![COHFACE Subject 1 vid 0](./imgs/COHFACE-1-0-save2.JPG)
#### COHFACE Subject 2 vid 0
![COHFACE Subject 2 vid 0](./imgs/COHFACE-2-0-save2.JPG)
#### UBFC2 Suject 1
![UBFC2 Suject 1](./imgs/UBFC-1-save2.JPG)
#### UBFC2 Subject 3
![UBFC2 Subject 3](./imgs/UBFC-3-save2.JPG)