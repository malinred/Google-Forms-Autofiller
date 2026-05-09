document.addEventListener('DOMContentLoaded', async () => {
    const profileSelect = document.getElementById('profileSelect');
    const autofillBtn = document.getElementById('autofillBtn');
    const statusDiv = document.getElementById('status');

    // Fetch available profiles from local server
    try {
        const response = await fetch('http://localhost:8000/profiles');
        const data = await response.json();

        profileSelect.innerHTML = '';
        if (data.profiles && data.profiles.length > 0) {
            data.profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile;
                option.textContent = profile;
                profileSelect.appendChild(option);
            });
            autofillBtn.disabled = false;
        } else {
            profileSelect.innerHTML = '<option value="">No profiles found</option>';
        }
    } catch (error) {
        console.error('Error fetching profiles:', error);
        statusDiv.textContent = 'Error: Could not connect to local server at localhost:8000. Make sure it is running.';
        statusDiv.style.color = '#c00';
    }

    autofillBtn.addEventListener('click', async () => {
        const selectedProfile = profileSelect.value;
        if (!selectedProfile) return;

        autofillBtn.disabled = true;
        statusDiv.style.color = '#666';
        statusDiv.textContent = 'Autofilling form...';

        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (!tab.url || !tab.url.includes('docs.google.com/forms')) {
                statusDiv.textContent = 'Error: Please navigate to a Google Form first.';
                statusDiv.style.color = '#c00';
                autofillBtn.disabled = false;
                return;
            }

            chrome.tabs.sendMessage(tab.id, {
                action: 'autofill',
                profileName: selectedProfile
            }, (response) => {
                autofillBtn.disabled = false;
                if (chrome.runtime.lastError) {
                    statusDiv.textContent = 'Error: ' + chrome.runtime.lastError.message;
                    statusDiv.style.color = '#c00';
                } else if (response && response.success) {
                    statusDiv.textContent = `Success! Filled ${response.filledCount} field(s).`;
                    statusDiv.style.color = '#1a7340';
                } else {
                    statusDiv.textContent = 'Autofill failed: ' + (response?.error || 'No fields matched.');
                    statusDiv.style.color = '#c00';
                }
            });
        } catch (error) {
            autofillBtn.disabled = false;
            statusDiv.textContent = 'Error: ' + error.message;
            statusDiv.style.color = '#c00';
        }
    });
});
