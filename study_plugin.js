/**
 * Study Plugin - Reading Behavior Tracking and AI Integration
 *
 * Tracks user reading patterns and sends observations to the AI assistant
 */

(function () {
    'use strict';

    /* ------------------------------------------------------------------
       Global state
    ------------------------------------------------------------------ */
    const state = {
        currentSection: null,
        sectionStartTime: null,
        lastScrollPosition: 0,
        scrollHistory: [],
        pauseTimer: null,
        lastInteractionTime: Date.now(),
        readingSpeeds: [],
        hoveredTerms: new Set(),
        interventionCount: 0,
        sectionInterventions: {},
        lastObservation: null,   
        sessionActive: true,      
        processingObservation: false,
        lastObservationHash: null,
        rereadCooldown: {},
        observationQueue: []  // ADD THIS: Queue for observations during AI processing
    };
    /* ------------------------------------------------------------------
       Config & templates
    ------------------------------------------------------------------ */
    const config = window.STUDY_CONFIG || {};
    const tracking = config.tracking || {};
    const templates = config.observation_templates || {};

    /* ------------------------------------------------------------------
       Utility functions
    ------------------------------------------------------------------ */
    function getCurrentSection() {
        const headers = document.querySelectorAll('h1, h2, h3');
        let currentSection = null;
        const currentY = window.scrollY + 100; // offset for better detection

        headers.forEach(header => {
            if (header.offsetTop <= currentY) currentSection = header;
        });

        return currentSection
            ? {
                  id: currentSection.id,
                  text: currentSection.textContent.trim(),
                  level: parseInt(currentSection.tagName[1]) 
              }
            : null;
    }

    /* -------- getVisibleContent --------------------------- */
    function getVisibleContent() {
        const viewportTop = window.scrollY;
        const viewportBottom = viewportTop + window.innerHeight;
        const elements = document.querySelectorAll(
            '.content-wrapper p, .content-wrapper li, ' +
                '.content-wrapper h1, .content-wrapper h2, .content-wrapper h3, ' +
                '.content-wrapper h4, .content-wrapper h5, .content-wrapper h6'
        );

        const visible = [];
        elements.forEach(el => {
            // Skip elements inside fixed / overlay containers
            if (
                el.closest('.toc-container') ||
                el.closest('.plugin-widget') ||
                el.closest('.control-panel')
            )
                return;

            const rect = el.getBoundingClientRect();
            const top = rect.top + window.scrollY;
            const bottom = rect.bottom + window.scrollY;

            if (top < viewportBottom && bottom > viewportTop) {
                visible.push({
                    element: el,
                    text: el.textContent.trim().substring(0, 100),
                    visibility: Math.min(
                        1,
                        (Math.min(bottom, viewportBottom) -
                            Math.max(top, viewportTop)) /
                            rect.height
                    )
                });
            }
        });

        if (!visible.length) {
            return [
                {
                    element: null,
                    text: 'document content',
                    visibility: 1
                }
            ];
        }
        return visible;
    }

    function calculateReadingSpeed() {
        if (state.scrollHistory.length < 2) return null;
        
        const now = Date.now();
        const recent = state.scrollHistory.slice(-5);
        
        // Check if user has been stationary (no scroll in last 2 seconds)
        const lastScrollTime = recent[recent.length - 1].time;
        if (now - lastScrollTime > 2000) {
            return 0; // User is reading, not scrolling
        }
        
        const distance = recent.reduce((sum, item, i) => {
            if (!i) return 0;
            return sum + Math.abs(item.position - recent[i - 1].position);
        }, 0);
        const time = recent[recent.length - 1].time - recent[0].time;
        
        // Avoid division by zero
        if (time === 0) return 0;
        
        return distance / (time / 1000); // px per second
    }

    function detectReReading(currentPosition) {
        const threshold = tracking.reread_detection_distance || 300; // Increased from 200
        const recent = state.scrollHistory.slice(-10);
        
        // Also check time - don't report re-reads too frequently
        const lastRereadTime = state.lastRereadTime || 0;
        const now = Date.now();
        
        if (now - lastRereadTime < 10000) { // 10 second minimum between re-reads
            return { detected: false };
        }
        
        for (let i = recent.length - 2; i >= 0; i--) {
            if (currentPosition < recent[i].position - threshold) {
                state.lastRereadTime = now; // Track when we detected re-read
                return {
                    detected: true,
                    distance: recent[i].position - currentPosition,
                    content: getVisibleContent()[0]?.text || 'unknown content'
                };
            }
        }
        return { detected: false };
    }

    /* ------------------------------------------------------------------
       UI helpers
    ------------------------------------------------------------------ */
    const messageHistory = [];

    function createLoadingElement() {
        const el = document.createElement('div');
        el.className = 'assistant-loading';
        el.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">
                <div class="loading-stage">Initializing...</div>
                <div class="loading-detail"></div>
            </div>
        `;
        document.getElementById('ai-response-container').appendChild(el);
        return el;
    }

    function updateLoadingState(stage, detail) {
        const s = document.querySelector('.loading-stage');
        const d = document.querySelector('.loading-detail');
        if (s) s.textContent = stage;
        if (d) d.textContent = detail;
    }

    /* -------- observation display----------------------- */
    function updateObservationDisplay() {
        let obs = document.getElementById('observation-display');
        if (!obs) {
            const header = document.querySelector('.assistant-header');
            if (!header) return;
            obs = document.createElement('div');
            obs.id = 'observation-display';
            obs.className = 'observation-display';
            header.insertAdjacentElement('afterend', obs);
        }

        if (state.lastObservation) {
            obs.innerHTML = `
                <div class="obs-label">Last observed:</div>
                <div class="obs-text">${state.lastObservation.text.substring(
                    0,
                    100
                )}...</div>
                <div class="obs-time">${state.lastObservation.time} - ${
                state.lastObservation.type
            }</div>
            `;
        }
    }

    /* -------- filterObservationsForAI --------------------------- */
    function filterObservationsForAI(observations) {
        // If 4 or fewer, send all
        if (observations.length <= 4) {
            return observations;
        }
        
        // Separate highlights and non-highlights
        const highlights = observations.filter(obs => obs.type === 'selection');
        const nonHighlights = observations.filter(obs => obs.type !== 'selection');
        
        let result = [];
        
        if (nonHighlights.length > 0) {
            // Keep the latest non-highlight
            const latestNonHighlight = nonHighlights[nonHighlights.length - 1];
            result.push(latestNonHighlight);
            
            if (highlights.length > 0) {
                // We have 3 slots left for highlights
                if (highlights.length <= 3) {
                    // Add all highlights
                    result.push(...highlights);
                } else {
                    // Apply "first two, last one" rule to highlights
                    result.push(highlights[0], highlights[1], highlights[highlights.length - 1]);
                }
                
                // If we still have less than 4, fill with latest non-highlights
                if (result.length < 4) {
                    const remainingSlots = 4 - result.length;
                    const otherNonHighlights = nonHighlights.slice(0, -1); // Exclude the one we already added
                    const latestOthers = otherNonHighlights.slice(-remainingSlots);
                    result.push(...latestOthers);
                }
            } else {
                // No highlights, just keep the latest non-highlights
                const latestNonHighlights = nonHighlights.slice(-3); // We already have 1
                result.push(...latestNonHighlights);
            }
        } else {
            // All highlights, apply "first two, last two" rule
            result.push(highlights[0], highlights[1], 
                    highlights[highlights.length - 2], highlights[highlights.length - 1]);
        }
        
        // Sort by timestamp to maintain chronological order
        result.sort((a, b) => a.timestamp - b.timestamp);
        
        return result;
    }

    /* -------- processQueuedObservations ------------------------- */
    async function processQueuedObservations() {
        if (state.observationQueue.length === 0) {
            console.log('[Study] No queued observations to process');
            return;
        }
        
        console.log(`[Study] Processing ${state.observationQueue.length} queued observations`);
        
        // Filter observations according to our rules
        const filteredObservations = filterObservationsForAI(state.observationQueue);
        
        console.log(`[Study] Filtered to ${filteredObservations.length} observations for AI`);
        
        // Clear the queue immediately
        state.observationQueue = [];
        
        // Process each observation
        for (const obs of filteredObservations) {
            // Don't update UI again, it was already updated when first observed
            console.log(`[Study] Sending queued observation: ${obs.type} from ${new Date(obs.timestamp).toLocaleTimeString()}`);
            
            // Reset processing state for each observation
            state.processingObservation = true;
            
            try {
                const response = await fetch('/api/plugin/reading-study/observe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        observation: obs.observation,
                        type: obs.type,
                        timestamp: obs.timestamp,
                        context: obs.context
                    })
                });
                
                const data = await response.json();
                
                if (data.response) {
                    displayAssistantResponse(data.response, data.type || 'suggestion');
                    state.interventionCount++;
                    if (state.currentSection) {
                        const id = state.currentSection.id;
                        state.sectionInterventions[id] = (state.sectionInterventions[id] || 0) + 1;
                    }
                }
            } catch (err) {
                console.error('[Study] Error processing queued observation:', err);
            }
            
            state.processingObservation = false;
        }
    }

    /* ------------------------------------------------------------------
       sendObservation
    ------------------------------------------------------------------ */
    async function sendObservation(observation, type = 'general') {
        // Ignore if session already ended
        if (!state.sessionActive) {
            console.log('[Study] Session ended, ignoring observation');
            return;
        }

        // Create observation hash to detect duplicates
        const observationHash = `${type}:${observation.substring(0, 50)}`;
        
        // Check if this is a duplicate of the last observation
        if (state.lastObservationHash === observationHash) {
            console.log('[Study] Duplicate observation detected, skipping');
            return;
        }

        // Special handling for re-read observations
        if (type === 'reread') {
            const now = Date.now();
            const cooldownKey = observation.substring(0, 30);
            
            if (state.rereadCooldown[cooldownKey] && 
                now - state.rereadCooldown[cooldownKey] < 60000) {
                console.log('[Study] Re-read in cooldown period, skipping');
                return;
            }
            
            state.rereadCooldown[cooldownKey] = now;
        }

        // ALWAYS update the display immediately
        state.lastObservation = {
            text: observation,
            type,
            time: new Date().toLocaleTimeString()
        };
        updateObservationDisplay();

        // Create observation object
        const obsData = {
            observation,
            type,
            timestamp: Date.now(),
            context: {
                section: state.currentSection?.text,
                scrollPosition: window.scrollY,
                readingSpeed: calculateReadingSpeed()
            }
        };

        // Check if already processing an observation
        if (state.processingObservation) {
            console.log('[Study] Already processing, queuing observation');
            state.observationQueue.push(obsData);
            return;
        }

        // Mark as processing
        state.processingObservation = true;
        state.lastObservationHash = observationHash;

        if (window.STUDY_MODE === 'testing') {
            console.log('[Study] Observation:', type, observation);
        }

        // Loading UI
        const container = document.getElementById('ai-response-container');
        if (container) {
            container.classList.add('active');
            const loading =
                container.querySelector('.assistant-loading') ||
                createLoadingElement();
            loading.style.display = 'flex';
            updateLoadingState(
                'Analyzing observation',
                'Understanding your reading pattern...'
            );
        }

        // Define stages for loading animation
        const stages = [
            { delay: 500, text: 'Analyzing observation', detail: 'Understanding your reading pattern...' },
            { delay: 1500, text: 'Inferring user state', detail: 'Assessing if you need help...' },
            { delay: 2500, text: 'Planning intervention', detail: 'Deciding how to assist...' },
            { delay: 3500, text: 'Generating response', detail: 'Creating helpful content...' }
        ];

        stages.forEach((stage, index) => {
            window[`stageTimeout${index}`] = setTimeout(() => {
                if (state.sessionActive) {
                    updateLoadingState(stage.text, stage.detail);
                }
            }, stage.delay);
        });

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const response = await fetch(
                '/api/plugin/reading-study/observe',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(obsData),
                    signal: controller.signal
                }
            );

            clearTimeout(timeoutId);

            // Clear any pending stage updates
            stages.forEach((stage, index) => {
                clearTimeout(window[`stageTimeout${index}`]);
            });

            const data = await response.json();

            if (container) {
                const loadingEl = container.querySelector('.assistant-loading');
                if (loadingEl) loadingEl.style.display = 'none';
            }

            if (data.response) {
                displayAssistantResponse(
                    data.response,
                    data.type || 'suggestion'
                );
                state.interventionCount++;
                if (state.currentSection) {
                    const id = state.currentSection.id;
                    state.sectionInterventions[id] =
                        (state.sectionInterventions[id] || 0) + 1;
                }
            }

            // Reset processing state on success
            state.processingObservation = false;
            
            // IMPORTANT: Process any queued observations
            await processQueuedObservations();
            
        } catch (err) {
            console.error('[Study] Error sending observation:', err);
            
            // Reset processing state on error
            state.processingObservation = false;
            
            // Clear any pending stage updates on error
            if (window.stageTimeout0) {
                for (let i = 0; i < 4; i++) {
                    clearTimeout(window[`stageTimeout${i}`]);
                }
            }

            if (container) {
                const loadingEl = container.querySelector('.assistant-loading');
                if (loadingEl) loadingEl.style.display = 'none';
            }

            if (err.name === 'AbortError') {
                console.log('[Study] Request timed out after 60 seconds');
                displayAssistantResponse(
                    'The AI assistant is taking longer than expected. Your reading activity is still being tracked.',
                    'info'
                );
            } else {
                updateLoadingState('Error', 'Failed to process observation');
            }
            
            // Still try to process queued observations even after error
            await processQueuedObservations();
        }
    }

    /* ------------------------------------------------------------------
       Assistant message panel (unchanged except guard checks)
    ------------------------------------------------------------------ */
    function displayAssistantResponse(response, type) {
        const container = document.getElementById('ai-response-container');
        if (!container) return;

        const msg = {
            id: Date.now(),
            response,
            type,
            timestamp: new Date().toLocaleTimeString(),
            feedbackProvided: false
        };
        messageHistory.push(msg);
        if (messageHistory.length > 5) messageHistory.shift();

        container.innerHTML = '';
        container.classList.add('active');

        const hideLoader = createLoadingElement();
        hideLoader.style.display = 'none';

        const list = document.createElement('div');
        list.className = 'message-history';

        messageHistory.forEach((m, i) => {
            const latest = i === messageHistory.length - 1;
            const el = document.createElement('div');
            el.className = `ai-response ${m.type} ${
                latest ? 'latest' : 'historical'
            }`;
            el.innerHTML = `
                <div class="response-header">
                    <span class="response-time">${m.timestamp}</span>
                    ${
                        m.feedbackProvided
                            ? '<span class="feedback-status">✓ Feedback provided</span>'
                            : ''
                    }
                </div>
                <div class="response-content">${m.response}</div>
                ${
                    latest && !m.feedbackProvided
                        ? `
                    <div class="response-actions">
                        <button class="feedback-btn helpful" onclick="provideFeedback(true, ${m.id})">
                            👍 Helpful
                        </button>
                        <button class="feedback-btn not-helpful" onclick="provideFeedback(false, ${m.id})">
                            👎 Not Helpful
                        </button>
                    </div>`
                        : ''
                }
            `;
            list.appendChild(el);
        });

        container.appendChild(list);
        list.scrollTop = list.scrollHeight;
    }

    window.provideFeedback = async function (helpful, messageId) {
        const msg = messageHistory.find(m => m.id === messageId);
        if (msg) msg.feedbackProvided = true;

        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.disabled = true;
            if (
                (helpful && btn.classList.contains('helpful')) ||
                (!helpful && btn.classList.contains('not-helpful'))
            )
                btn.classList.add('clicked');
        });

        await fetch('/api/plugin/reading-study/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ helpful })
        });

        setTimeout(() => {
            if (messageHistory.length) displayAssistantResponse('', '');
        }, 500);
    };

    window.dismissResponse = function () {
        const c = document.getElementById('ai-response-container');
        if (c) c.classList.remove('active');
    };

    /* ------------------------------------------------------------------
       Event handlers (all respect sessionActive)
    ------------------------------------------------------------------ */
    function handlePause() {
        if (!state.sessionActive) return;

        const now = Date.now();
        const idle = now - state.lastInteractionTime;

        if (idle > (tracking.min_pause_duration || 3000)) {
            const visible = getVisibleContent();
            if (visible.length) {
                const dur = Math.round(idle / 1000);
                sendObservation(
                    templates.pause
                        .replace('{{content}}', visible[0].text)
                        .replace('{{duration}}', dur),
                    'pause'
                );
            }
        }
    }

    function handleMouseMove(e) {
        if (!state.sessionActive) return;

        state.lastInteractionTime = Date.now();
        if (state.pauseTimer) clearTimeout(state.pauseTimer);
        state.pauseTimer = setTimeout(
            handlePause,
            tracking.min_pause_duration || 3000
        );
    }

    function handleTextSelection() {
        if (!state.sessionActive) return;

        const text = window.getSelection().toString().trim();
        if (text.length > 5) {
            sendObservation(
                templates.selection.replace('{{text}}', text.substring(0, 100)),
                'selection'
            );
        }
    }

    function handleHover(e) {
        if (!state.sessionActive) return;

        const t = e.target;
        if (
            t.tagName === 'CODE' ||
            t.classList.contains('term') ||
            t.tagName === 'EM' ||
            t.tagName === 'STRONG'
        ) {
            const term = t.textContent.trim();
            if (term && !state.hoveredTerms.has(term)) {
                state.hoveredTerms.add(term);
                setTimeout(() => {
                    if (!state.sessionActive) return;
                    if (t.matches(':hover')) {
                        sendObservation(
                            templates.hover.replace('{{term}}', term),
                            'hover'
                        );
                    }
                }, tracking.hover_detection_delay || 1000);
            }
        }
    }

    function handleVisibilityChange() {
        if (!state.sessionActive) return;

        if (document.hidden) {
            sendObservation(templates.focus_lost, 'focus_lost');
            state.focusLostTime = Date.now();
        } else if (state.focusLostTime) {
            const dur = Math.round((Date.now() - state.focusLostTime) / 1000);
            sendObservation(
                templates.focus_return.replace('{{duration}}', dur),
                'focus_return'
            );
        }
    }

    /* -------- scroll handler impl  ------------------------- */
    const handleScrollImpl = function () {
        if (!state.sessionActive) return;

        const pos = window.scrollY;
        const now = Date.now();

        state.scrollHistory.push({ position: pos, time: now });
        if (state.scrollHistory.length > 50) state.scrollHistory.shift();

        const newSection = getCurrentSection();
        if (
            newSection &&
            (!state.currentSection || newSection.id !== state.currentSection.id)
        ) {
            if (state.currentSection && state.sectionStartTime) {
                const dur = Math.round((now - state.sectionStartTime) / 1000);
                sendObservation(
                    templates.section_complete
                        .replace('{{section}}', state.currentSection.text)
                        .replace('{{duration}}', dur),
                    'section_complete'
                );
            }
            state.currentSection = newSection;
            state.sectionStartTime = now;
            sendObservation(
                templates.section_start.replace('{{section}}', newSection.text),
                'section_start'
            );
        }

        const rr = detectReReading(pos);
        if (rr.detected) {
            sendObservation(
                templates.reread.replace('{{topic}}', rr.content),
                'reread'
            );
        }

        const speed = calculateReadingSpeed();
        if (
            speed &&
            speed > (config.metrics?.reading_speed_thresholds?.fast || 400)
        ) {
            const v = getVisibleContent();
            if (v.length) {
                sendObservation(
                    templates.rapid_scroll.replace('{{content}}', v[0].text),
                    'rapid_scroll'
                );
            }
        }

        state.lastScrollPosition = pos;
        state.lastInteractionTime = now;
    };

    /* ------------------------------------------------------------------
       Session finish
    ------------------------------------------------------------------ */
    window.endStudySession = async function () {
        if (!state.sessionActive) {
            alert('Session has already ended.');
            return;
        }

        if (!confirm('Are you sure you want to end the study session?')) {
            return;
        }

        state.sessionActive = false;
        stopAllTracking();

        try {
            const res = await fetch('/api/plugin/reading-study/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: 'end_session' })
            });
            const data = await res.json();

            if (data.status === 'completed') {
                const widget =
                    document.getElementById('ai-assistant-widget') ||
                    document.getElementById('ai-response-container');
                if (widget) {
                    widget.innerHTML = `
                        <div class="session-ended">
                            <h3>✅ Study Session Complete</h3>
                            <p>Thank you for participating!</p>
                            <p>Session data saved to:</p>
                            <code>${data.filename}</code>
                            <p style="margin-top: 20px;">You can now close this window.</p>
                            <button onclick="window.close()" style="margin-top: 10px; padding: 8px 16px;">
                                Close Window
                            </button>
                        </div>
                    `;
                }
                alert(data.message || 'Study session completed. Thank you!');
            }
        } catch (err) {
            console.error('Error ending session:', err);
            alert('Session ended locally. You can close the window.');
        }
    };

    /* -------- stopAllTracking ----------------------------- */
    function stopAllTracking() {
        window.removeEventListener('scroll', window.handleScroll);
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleTextSelection);
        document.removeEventListener('mouseover', handleHover);
        document.removeEventListener('visibilitychange', handleVisibilityChange);

        if (state.pauseTimer) clearTimeout(state.pauseTimer);
        if (window.scrollTimer) clearTimeout(window.scrollTimer);

        console.log('[Study] All tracking stopped');
    }

    /* ------------------------------------------------------------------
       Initialise tracking
    ------------------------------------------------------------------ */
    function initializeTracking() {
        // Named handler so we can remove it later
        window.handleScroll = () => {
            if (!state.sessionActive) return;
            clearTimeout(window.scrollTimer);
            window.scrollTimer = setTimeout(
                () =>
                    state.sessionActive &&
                    handleScrollImpl(),
                tracking.scroll_sample_rate || 100
            );
        };

        window.addEventListener('scroll', window.handleScroll);
        document.addEventListener('mousemove', handleMouseMove);

        if (tracking.selection_tracking !== false) {
            document.addEventListener('mouseup', handleTextSelection);
        }
        document.addEventListener('mouseover', handleHover);

        if (tracking.focus_tracking !== false) {
            document.addEventListener(
                'visibilitychange',
                handleVisibilityChange
            );
        }

        state.currentSection = getCurrentSection();
        state.sectionStartTime = Date.now();

        // Debug overlay (unchanged)
        if (window.STUDY_MODE === 'testing') {
            console.log('[Study] Tracking initialized');
            console.log('[Study] Config:', config);

            const dbg = document.createElement('div');
            dbg.id = 'study-debug';
            dbg.style.cssText = `
                position: fixed;
                bottom: 10px;
                left: 10px;
                background: rgba(0,0,0,0.8);
                color: white;
                padding: 10px;
                font-size: 12px;
                font-family: monospace;
                z-index: 10000;
            `;
            document.body.appendChild(dbg);

            setInterval(() => {
                const speed = calculateReadingSpeed();
                const speedDisplay = speed === null ? 'N/A' : 
                                    speed === 0 ? 'Stationary' : 
                                    `${speed.toFixed(0)} px/s`;
                
                dbg.innerHTML = `
                    Mode: ${window.STUDY_MODE}<br>
                    Section: ${state.currentSection?.text || 'None'}<br>
                    Interventions: ${state.interventionCount}<br>
                    Reading Speed: ${speedDisplay}
                `;
            }, 1000);
        }
    }

    /* ------------------------------------------------------------------
       DOM ready
    ------------------------------------------------------------------ */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeTracking);
    } else {
        initializeTracking();
    }
})();
