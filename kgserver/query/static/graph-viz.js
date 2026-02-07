/**
 * Knowledge Graph Force-Directed Visualization
 * Uses D3.js v7 for rendering
 */

// State
let simulation = null;
let svg = null;
let g = null;  // Main group for zoom/pan
let zoom = null;  // Zoom behavior reference
let currentData = { nodes: [], edges: [] };

// DOM Elements
const entitySearchInput = document.getElementById('entity-search');
const searchResultsContainer = document.getElementById('search-results');
const centerIdInput = document.getElementById('center-id');
const hopsInput = document.getElementById('hops');
const hopsValue = document.getElementById('hops-value');
const maxNodesInput = document.getElementById('max-nodes');
const maxNodesValue = document.getElementById('max-nodes-value');
const includeAllCheckbox = document.getElementById('include-all');
const loadButton = document.getElementById('load-graph');
const resetButton = document.getElementById('reset-view');
const tooltip = document.getElementById('tooltip');
const detailPanel = document.getElementById('detail-panel');
const panelTitle = document.getElementById('panel-title');
const panelContent = document.getElementById('panel-content');
const closePanel = document.getElementById('close-panel');

// Search state
let searchTimeout = null;
let selectedSearchIndex = -1;

// Stats elements
const nodeCountEl = document.getElementById('node-count');
const edgeCountEl = document.getElementById('edge-count');
const totalEntitiesEl = document.getElementById('total-entities');
const totalRelationshipsEl = document.getElementById('total-relationships');
const truncatedWarning = document.getElementById('truncated-warning');

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
    setupSVG();
    setupEventListeners();

    // Check URL params for initial load
    const params = new URLSearchParams(window.location.search);
    const centerId = params.get('center_id');
    if (centerId) {
        centerIdInput.value = centerId;
        loadGraph();
    } else {
        // On first visit with no params, show entire graph
        includeAllCheckbox.checked = true;
        centerIdInput.disabled = true;
        entitySearchInput.disabled = true;
        loadGraph();
    }
}

function setupSVG() {
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    svg = d3.select('#graph-svg')
        .attr('viewBox', [0, 0, width, height]);
    
    // Add arrow marker definition
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('path')
        .attr('d', 'M 0,-5 L 10,0 L 0,5')
        .attr('fill', '#999');
    
    // Create main group for zoom/pan
    g = svg.append('g');
    
    // Setup zoom behavior
    zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    // Handle window resize
    window.addEventListener('resize', () => {
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        svg.attr('viewBox', [0, 0, newWidth, newHeight]);
        if (simulation) {
            simulation.force('center', d3.forceCenter(newWidth / 2, newHeight / 2));
            simulation.alpha(0.3).restart();
        }
    });
}

function setupEventListeners() {
    // Search input with debounce
    entitySearchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        // Clear previous timeout
        if (searchTimeout) clearTimeout(searchTimeout);
        
        if (query.length < 2) {
            hideSearchResults();
            return;
        }
        
        // Debounce: wait 300ms after typing stops
        searchTimeout = setTimeout(() => {
            performSearch(query);
        }, 300);
    });
    
    // Keyboard navigation for search results
    entitySearchInput.addEventListener('keydown', (e) => {
        const items = searchResultsContainer.querySelectorAll('.search-result-item');
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectedSearchIndex = Math.min(selectedSearchIndex + 1, items.length - 1);
            updateSearchSelection(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectedSearchIndex = Math.max(selectedSearchIndex - 1, 0);
            updateSearchSelection(items);
        } else if (e.key === 'Enter' && selectedSearchIndex >= 0) {
            e.preventDefault();
            items[selectedSearchIndex]?.click();
        } else if (e.key === 'Escape') {
            hideSearchResults();
        }
    });
    
    // Hide search results when clicking outside
    document.addEventListener('click', (e) => {
        if (!entitySearchInput.contains(e.target) && !searchResultsContainer.contains(e.target)) {
            hideSearchResults();
        }
    });
    
    // Update display values for range inputs
    hopsInput.addEventListener('input', () => {
        hopsValue.textContent = hopsInput.value;
    });
    
    maxNodesInput.addEventListener('input', () => {
        maxNodesValue.textContent = maxNodesInput.value;
    });
    
    // Toggle center_id input based on include_all checkbox
    includeAllCheckbox.addEventListener('change', () => {
        centerIdInput.disabled = includeAllCheckbox.checked;
        entitySearchInput.disabled = includeAllCheckbox.checked;
    });
    
    // Load graph button
    loadButton.addEventListener('click', loadGraph);
    
    // Enter key in center_id input
    centerIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadGraph();
    });
    
    // Reset view button
    resetButton.addEventListener('click', resetView);
    
    // Close detail panel
    closePanel.addEventListener('click', () => {
        detailPanel.classList.add('hidden');
    });
}

// Search functionality
async function performSearch(query) {
    showSearchLoading();
    
    try {
        const response = await fetch(`/api/v1/graph/search?q=${encodeURIComponent(query)}&limit=20`);
        if (!response.ok) {
            throw new Error('Search failed');
        }
        
        const data = await response.json();
        renderSearchResults(data.results, data.total);
        
    } catch (error) {
        console.error('Search error:', error);
        searchResultsContainer.innerHTML = '<div class="search-no-results">Search failed</div>';
        searchResultsContainer.classList.remove('hidden');
    }
}

function renderSearchResults(results, total) {
    selectedSearchIndex = -1;
    
    if (results.length === 0) {
        searchResultsContainer.innerHTML = '<div class="search-no-results">No entities found</div>';
        searchResultsContainer.classList.remove('hidden');
        return;
    }
    
    let html = results.map((result, index) => `
        <div class="search-result-item" data-entity-id="${escapeHtml(result.entity_id)}" data-index="${index}">
            <div class="search-result-name">${escapeHtml(result.name)}</div>
            <div class="search-result-type">${escapeHtml(result.entity_type)}</div>
            <div class="search-result-id">${escapeHtml(result.entity_id)}</div>
        </div>
    `).join('');
    
    if (total > results.length) {
        html += `<div class="search-no-results">${total - results.length} more results...</div>`;
    }
    
    searchResultsContainer.innerHTML = html;
    searchResultsContainer.classList.remove('hidden');
    
    // Add click handlers
    searchResultsContainer.querySelectorAll('.search-result-item').forEach(item => {
        item.addEventListener('click', () => {
            const entityId = item.dataset.entityId;
            selectSearchResult(entityId);
        });
    });
}

function selectSearchResult(entityId) {
    centerIdInput.value = entityId;
    entitySearchInput.value = '';
    hideSearchResults();
    includeAllCheckbox.checked = false;
    centerIdInput.disabled = false;
    entitySearchInput.disabled = false;
    loadGraph();
}

function updateSearchSelection(items) {
    items.forEach((item, i) => {
        item.classList.toggle('selected', i === selectedSearchIndex);
    });
    
    // Scroll selected item into view
    if (selectedSearchIndex >= 0 && items[selectedSearchIndex]) {
        items[selectedSearchIndex].scrollIntoView({ block: 'nearest' });
    }
}

function showSearchLoading() {
    searchResultsContainer.innerHTML = '<div class="search-loading">Searching...</div>';
    searchResultsContainer.classList.remove('hidden');
}

function hideSearchResults() {
    searchResultsContainer.classList.add('hidden');
    selectedSearchIndex = -1;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadGraph() {
    const includeAll = includeAllCheckbox.checked;
    const centerId = centerIdInput.value.trim();
    const hops = parseInt(hopsInput.value);
    const maxNodes = parseInt(maxNodesInput.value);
    
    if (!includeAll && !centerId) {
        showError('Please enter a center entity ID or check "Show entire graph"');
        return;
    }
    
    // Build URL
    let url = `/api/v1/graph/subgraph?hops=${hops}&max_nodes=${maxNodes}`;
    if (includeAll) {
        url += '&include_all=true';
    } else {
        url += `&center_id=${encodeURIComponent(centerId)}`;
    }
    
    // Show loading state
    showLoading();
    
    try {
        const response = await fetch(url);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load graph');
        }
        
        const data = await response.json();
        currentData = data;
        
        // Update stats
        updateStats(data);
        
        // Render graph
        renderGraph(data);
        
    } catch (error) {
        showError(error.message);
    }
}

function updateStats(data) {
    nodeCountEl.textContent = data.nodes.length;
    edgeCountEl.textContent = data.edges.length;
    totalEntitiesEl.textContent = data.total_entities;
    totalRelationshipsEl.textContent = data.total_relationships;
    
    if (data.truncated) {
        truncatedWarning.classList.remove('hidden');
    } else {
        truncatedWarning.classList.add('hidden');
    }
}

function renderGraph(data) {
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear existing content
    g.selectAll('*').remove();
    
    if (data.nodes.length === 0) {
        showError('No nodes found for the given criteria');
        return;
    }
    
    // Create links data with source/target references
    const links = data.edges.map(d => ({
        ...d,
        source: d.source,
        target: d.target
    }));
    
    // Create nodes data
    const nodes = data.nodes.map(d => ({ ...d }));
    
    // Create simulation
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links)
            .id(d => d.id)
            .distance(150))
        .force('charge', d3.forceManyBody()
            .strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
            .radius(40));
    
    // Create link elements
    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke', '#999')
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)')
        .on('click', (event, d) => showEdgeDetails(d))
        .on('mouseenter', (event, d) => showTooltip(event, `${d.label}`))
        .on('mouseleave', hideTooltip);
    
    // Create link labels
    const linkLabel = g.append('g')
        .attr('class', 'link-labels')
        .selectAll('text')
        .data(links)
        .join('text')
        .attr('class', 'link-label')
        .text(d => d.label);
    
    // Create node elements
    const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', d => `node node-${getNodeTypeClass(d.entity_type)}`)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended))
        .on('click', (event, d) => showNodeDetails(d))
        .on('dblclick', (event, d) => recenterOn(d.id))
        .on('mouseenter', (event, d) => showTooltip(event, `${d.label}\n(${d.entity_type})`))
        .on('mouseleave', hideTooltip);
    
    // Add circles to nodes
    node.append('circle')
        .attr('r', d => d.id === data.center_id ? 12 : 8);
    
    // Add labels to nodes
    node.append('text')
        .attr('dy', 20)
        .text(d => truncateLabel(d.label, 15));
    
    // Update positions on tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        linkLabel
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2);
        
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

function getNodeTypeClass(entityType) {
    const type = (entityType || '').toLowerCase();
    const knownTypes = ['disease', 'drug', 'gene', 'protein', 'procedure', 'symptom', 'anatomy'];
    return knownTypes.includes(type) ? type : 'default';
}

function truncateLabel(label, maxLength) {
    if (!label) return '';
    return label.length > maxLength ? label.substring(0, maxLength) + '...' : label;
}

// Drag handlers
function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Tooltip
function showTooltip(event, text) {
    tooltip.textContent = text;
    tooltip.classList.remove('hidden');
    tooltip.style.left = (event.pageX + 10) + 'px';
    tooltip.style.top = (event.pageY + 10) + 'px';
}

function hideTooltip() {
    tooltip.classList.add('hidden');
}

// Detail panel
function showNodeDetails(node) {
    panelTitle.textContent = node.label || node.id;
    
    let html = '';
    const props = node.properties || {};
    
    html += createProperty('Entity ID', props.entity_id);
    html += createProperty('Type', props.entity_type);
    html += createProperty('Name', props.name);
    html += createProperty('Status', props.status);
    html += createProperty('Confidence', props.confidence);
    html += createProperty('Usage Count', props.usage_count);
    
    // Make source a clickable link if it's a PMC ID
    if (props.source) {
        const sourceLink = formatSourceLink(props.source);
        html += createProperty('Source', sourceLink);
    }
    
    if (props.canonical_url) {
        html += createProperty('Canonical URL', 
            `<a href="${props.canonical_url}" target="_blank">${props.canonical_url}</a>`);
    }
    
    if (props.synonyms && props.synonyms.length > 0) {
        html += createProperty('Synonyms', props.synonyms.join(', '));
    }
    
    panelContent.innerHTML = html;
    detailPanel.classList.remove('hidden');
}

/**
 * Format a source identifier as a clickable link when possible.
 * Supports PMC IDs (PubMed Central) and PMID (PubMed).
 */
function formatSourceLink(source) {
    if (!source) return '';
    
    // PMC ID format: PMC followed by digits (e.g., PMC10759991)
    const pmcMatch = source.match(/^PMC(\d+)$/i);
    if (pmcMatch) {
        const pmcUrl = `https://www.ncbi.nlm.nih.gov/pmc/articles/${source}/`;
        return `<a href="${pmcUrl}" target="_blank" title="View on PubMed Central">${source}</a>`;
    }
    
    // PMID format: just digits or "PMID:" prefix
    const pmidMatch = source.match(/^(?:PMID[:\s]*)?(\d{6,9})$/i);
    if (pmidMatch) {
        const pmid = pmidMatch[1];
        const pmidUrl = `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
        return `<a href="${pmidUrl}" target="_blank" title="View on PubMed">PMID:${pmid}</a>`;
    }
    
    // DOI format
    if (source.startsWith('10.') || source.toLowerCase().startsWith('doi:')) {
        const doi = source.replace(/^doi:/i, '');
        const doiUrl = `https://doi.org/${doi}`;
        return `<a href="${doiUrl}" target="_blank" title="View via DOI">${source}</a>`;
    }
    
    // Return as-is if no pattern matches
    return source;
}

function showEdgeDetails(edge) {
    panelTitle.textContent = edge.label;
    
    let html = '';
    const props = edge.properties || {};
    
    html += createProperty('Predicate', props.predicate);
    html += createProperty('Subject', props.subject_id);
    html += createProperty('Object', props.object_id);
    html += createProperty('Confidence', props.confidence);
    
    // Make source documents clickable links
    if (props.source_documents && props.source_documents.length > 0) {
        const linkedDocs = props.source_documents
            .map(doc => formatSourceLink(doc))
            .join(', ');
        html += createProperty('Source Documents', linkedDocs);
    }
    
    panelContent.innerHTML = html;
    detailPanel.classList.remove('hidden');
}

function createProperty(label, value) {
    if (value === null || value === undefined || value === '') return '';
    return `
        <div class="property">
            <div class="property-label">${label}</div>
            <div class="property-value">${value}</div>
        </div>
    `;
}

// Recenter graph on a node
function recenterOn(entityId) {
    centerIdInput.value = entityId;
    includeAllCheckbox.checked = false;
    centerIdInput.disabled = false;
    loadGraph();
}

// Reset view - fit graph to window at 90% with centering
function resetView() {
    if (!simulation || currentData.nodes.length === 0) {
        // No graph loaded, just reset to identity
        svg.transition()
            .duration(750)
            .call(zoom.transform, d3.zoomIdentity);
        return;
    }
    
    const container = document.getElementById('graph-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Calculate bounding box of all nodes
    const nodes = simulation.nodes();
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    nodes.forEach(node => {
        if (node.x < minX) minX = node.x;
        if (node.x > maxX) maxX = node.x;
        if (node.y < minY) minY = node.y;
        if (node.y > maxY) maxY = node.y;
    });
    
    // Add padding for node radius and labels
    const padding = 50;
    minX -= padding;
    maxX += padding;
    minY -= padding;
    maxY += padding;
    
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;
    
    // Calculate scale to fit graph in 90% of viewport
    const scaleX = (width * 0.9) / graphWidth;
    const scaleY = (height * 0.9) / graphHeight;
    const scale = Math.min(scaleX, scaleY, 2);  // Cap at 2x zoom
    
    // Calculate center of the graph
    const graphCenterX = (minX + maxX) / 2;
    const graphCenterY = (minY + maxY) / 2;
    
    // Calculate translation to center the graph
    const translateX = width / 2 - graphCenterX * scale;
    const translateY = height / 2 - graphCenterY * scale;
    
    // Apply transform with smooth transition
    const transform = d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale);
    
    svg.transition()
        .duration(750)
        .call(zoom.transform, transform);
}

// Show loading state
function showLoading() {
    g.selectAll('*').remove();
    g.append('text')
        .attr('class', 'loading')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('fill', '#888')
        .text('Loading...');
}

// Show error
function showError(message) {
    g.selectAll('*').remove();
    g.append('text')
        .attr('class', 'error-message')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('fill', '#e94560')
        .text(message);
}
