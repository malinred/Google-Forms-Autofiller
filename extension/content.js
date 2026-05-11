/**
 * Content script for Google Forms Autofiller
 */

// Native setter bypass for React-controlled inputs
const _nativeInputSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
const _nativeTextareaSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'autofill') {
        autofillForm(request.profileName)
            .then(result => sendResponse(result))
            .catch(err => sendResponse({ success: false, error: err.message }));
        return true;
    }
});

async function autofillForm(profileName) {
    console.log('[Autofiller] Starting autofill for profile:', profileName);

    const response = await fetch(`http://localhost:8000/profiles/${profileName}`);
    if (!response.ok) throw new Error('Failed to fetch profile data from server');
    const profileData = await response.json();
    console.log('[Autofiller] Profile data loaded:', Object.keys(profileData));

    await new Promise(resolve => setTimeout(resolve, 500));

    const questions = document.querySelectorAll(
        'div[role="listitem"], div.Qr7Oae, div.freebirdFormviewerViewItemsItemItem'
    );

    if (questions.length === 0) {
        console.warn('[Autofiller] No form questions found on this page.');
        return { success: false, filledCount: 0, error: 'No questions found' };
    }

    console.log(`[Autofiller] Found ${questions.length} questions`);
    let filledCount = 0;

    for (const q of questions) {
        const labelEl = q.querySelector(
            'div[role="heading"] span.M7eMe, div[role="heading"], span.M7eMe, div.freebirdFormviewerViewItemsItemItemTitle'
        );
        if (!labelEl) continue;

        const questionText = labelEl.innerText.trim();
        if (!questionText) continue;

        console.log('[Autofiller] Processing question:', questionText);

        const matchedValue = findBestMatch(questionText, profileData);
        if (!matchedValue) {
            console.log('[Autofiller] No match found for:', questionText);
            continue;
        }

        console.log('[Autofiller] Matched value:', matchedValue);

        // --- 1. Text / Email / Tel / Number / Textarea ---
        const textInput = q.querySelector(
            'input[type="text"], input[type="email"], input[type="tel"], input[type="number"], textarea'
        );
        if (textInput) {
            console.log('[Autofiller] Filling text input');
            fillInput(textInput, matchedValue);
            filledCount++;
            continue;
        }

        // --- 2. Radio buttons (Google Forms actual DOM) ---
        // Real selector from live Google Forms DOM: div[role="radio"][data-value]
        const radioOptions = q.querySelectorAll('div[role="radio"][data-value]');
        if (radioOptions.length > 0) {
            console.log(`[Autofiller] Found ${radioOptions.length} radio options`);
            let matched = false;
            for (const radio of radioOptions) {
                const radioValue = radio.getAttribute('data-value') || '';
                const ariaLabel = radio.getAttribute('aria-label') || '';
                if (
                    radioValue.toLowerCase() === matchedValue.toLowerCase() ||
                    ariaLabel.toLowerCase() === matchedValue.toLowerCase() ||
                    matchedValue.toLowerCase().includes(radioValue.toLowerCase()) ||
                    radioValue.toLowerCase().includes(matchedValue.toLowerCase())
                ) {
                    radio.click();
                    filledCount++;
                    matched = true;
                    console.log('[Autofiller] Clicked radio:', radioValue);
                    break;
                }
            }
            if (matched) continue;
        }

        // --- 3. Checkboxes (Google Forms actual DOM) ---
        // Real selector: div[role="checkbox"][data-value]
        const checkboxOptions = q.querySelectorAll('div[role="checkbox"][data-value]');
        if (checkboxOptions.length > 0) {
            console.log(`[Autofiller] Found ${checkboxOptions.length} checkbox options`);
            const valuesToCheck = matchedValue.split(',').map(v => v.trim().toLowerCase());
            for (const checkbox of checkboxOptions) {
                const checkboxValue = (checkbox.getAttribute('data-value') || '').toLowerCase();
                const ariaLabel = (checkbox.getAttribute('aria-label') || '').toLowerCase();
                if (valuesToCheck.some(v => v === checkboxValue || v === ariaLabel)) {
                    const alreadyChecked = checkbox.getAttribute('aria-checked') === 'true';
                    if (!alreadyChecked) {
                        checkbox.click();
                        filledCount++;
                        console.log('[Autofiller] Clicked checkbox:', checkboxValue);
                    }
                }
            }
            continue;
        }

        // --- 4. Dropdown (select element) ---
        const dropdown = q.querySelector('select');
        if (dropdown) {
            console.log('[Autofiller] Filling dropdown');
            for (const option of dropdown.options) {
                if (option.text.toLowerCase() === matchedValue.toLowerCase()) {
                    dropdown.value = option.value;
                    dropdown.dispatchEvent(new Event('change', { bubbles: true }));
                    filledCount++;
                    break;
                }
            }
            continue;
        }
    }

    console.log(`[Autofiller] Done. Filled ${filledCount} fields.`);
    return { success: true, filledCount };
}

function findBestMatch(questionText, profileData) {
    const normalizedQuestion = questionText.toLowerCase().trim();

    // 1. Direct key match (case-insensitive, strip trailing colon)
    for (const [key, value] of Object.entries(profileData)) {
        const normalizedKey = key.toLowerCase().replace(/:$/, '').trim();
        if (normalizedQuestion === normalizedKey) return value;
        if (normalizedQuestion.includes(normalizedKey) && normalizedKey.length > 3) return value;
        if (normalizedKey.includes(normalizedQuestion) && normalizedQuestion.length > 3) return value;
    }

    // 2. Semantic keyword mappings
    const semanticMap = {
        'name':               ['name'],
        'full name':          ['name'],
        'gender':             ['gender'],
        'nationality':        ['nationality'],
        'blood':              ['blood group'],
        'marital':            ['marital status'],
        'religion':           ['religion'],
        'mother tongue':      ['mother tongue'],
        'mobile':             ['mobile number'],
        'phone':              ['mobile number'],
        'alternate.*number':  ['alternate number'],
        'email':              ['email address'],
        'alternate.*email':   ['alternate email'],
        'address':            ['permanent address'],
        'city':               ['city'],
        'state':              ['state'],
        'pin':                ['pin code'],
        'postal':             ['pin code'],
        'zip':                ['pin code'],
        'country':            ['country'],
        'institution':        ['institution'],
        'college':            ['institution'],
        'university':         ['institution'],
        'department':         ['school of computing'],
        'course':             ['school of computing'],
        'roll':               ['of technology'],
        'register':           ['of technology'],
        'joining':            ['year of joining'],
        'passing':            ['year of passing'],
        'graduation':         ['year of passing'],
        'semester':           ['current semester'],
        'cgpa':               ['cgpa'],
        'gpa':                ['cgpa'],
        'aadhaar':            ['aadhaar number'],
        'aadhar':             ['aadhaar number'],
        'pan':                ['pan number'],
        'passport':           ['passport number'],
        'passport expiry':    ['passport expiry'],
        'voter':              ['voter id'],
        'driving':            ['driving license'],
        'licence':            ['driving license'],
        'account.*holder':    ['account holder name'],
        'account.*name':      ['account holder name'],
        'bank.*name':         ['state bank'],
        'bank.*account':      ['mg road'],
        'ifsc':               ['mg road'],
        'account.*number':    ['mg road'],
        'emergency.*name':    ['contact name'],
        'emergency.*contact': ['contact name'],
        'relationship':       ['relationship'],
        'relation':           ['relationship'],
    };

    for (const [pattern, profileKeys] of Object.entries(semanticMap)) {
        const regex = new RegExp(pattern, 'i');
        if (regex.test(normalizedQuestion)) {
            for (const pKey of profileKeys) {
                const actualKey = Object.keys(profileData).find(k =>
                    k.toLowerCase().replace(/:$/, '').includes(pKey.toLowerCase())
                );
                if (actualKey) return profileData[actualKey];
            }
        }
    }

    return null;
}

function fillInput(input, value) {
    input.focus();
    const isTextarea = input.tagName.toLowerCase() === 'textarea';
    const setter = isTextarea ? _nativeTextareaSetter : _nativeInputSetter;

    if (setter) setter.call(input, value);
    else input.value = value;

    input.dispatchEvent(new InputEvent('input', {
        bubbles: true,
        cancelable: true,
        inputType: 'insertText',
        data: value
    }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
    input.dispatchEvent(new FocusEvent('blur', { bubbles: true }));
    input.blur();
}
