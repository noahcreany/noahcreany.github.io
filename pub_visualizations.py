#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 15:02:36 2026

@author: noahcreany
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from pybtex.database.input import bibtex
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from parsel import Selector
import requests, os, json
import time
import random
import os
from datetime import datetime

# --- Configuration ---
BIB_FILE = '/Volumes/PNY_USB/Professional/quarto-cv/sections/biblio.bib'
SIMILARITY_THRESHOLD = 0.4
JOURNAL_OUTPUT = '_includes/journal_counts.html'
KEYWORD_OUTPUT = '_includes/keyword_network.html'
PLOT_WIDTH = 550
PLOT_HEIGHT = 400

def get_color_palette(n_shades):
    """
    Provides color shades for bar plots.
    """
    # Base palette
    base_colors = ['#264653','#287271','#2a9d8f','#8ab17d','#e9c46a','#efb366','#f4a261','#ee8959','#e76f51']
    
    if n_shades <= len(base_colors):
        return base_colors[:n_shades]
    else:
        # If more shades needed, interpolate or repeat. For now, repeat/cycle
        return (base_colors * (n_shades // len(base_colors) + 1))[:n_shades]

from datetime import datetime

def load_bibliography(file_path):
    parser = bibtex.Parser()
    return parser.parse_file(file_path)

def generate_journal_plot(bib_data):
    print("Generating Journal Plot")
    
    try:
        alt.theme.enable('urbaninstitute')
    except:
        # If strict theme fails, we manually apply the key styles observed:
        # - Typography: Arial usually.
        # - Clean axes.
        pass

    journals = []
    for entry in bib_data.entries.values():
        if 'journal' in entry.fields:
            journals.append(entry.fields['journal'])
    
    if not journals:
        print("No journals found.")
        return

    df = pd.DataFrame({'Journal': journals})
    df_counts = df['Journal'].value_counts().reset_index()
    df_counts.columns = ['Journal', 'Count']
    
    # Text wrapping helper
    def wrap_labels(label, max_len=25):
        words = label.split()
        lines = []
        current_line = []
        current_len = 0
        for word in words:
            if current_len + len(word) <= max_len:
                current_line.append(word)
                current_len += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
        if current_line:
            lines.append(" ".join(current_line))
        return lines # Return list for Altair multi-line labels

    df_counts['Journal_Wrapped'] = df_counts['Journal'].apply(lambda x: wrap_labels(x))
    
    # Get palette
    n_journals = len(df_counts)
    palette = get_color_palette(n_journals)
    
    today = datetime.now().strftime('%-d %b %Y')
    
    # Altair Chart
    base = alt.Chart(df_counts).encode(
        y=alt.Y('Journal_Wrapped', sort=alt.EncodingSortField(field="Count", order="descending"), axis=alt.Axis(title=None, labelFontSize=13)),
        x=alt.X('Count', axis=alt.Axis(title=None, tickMinStep=1, tickCount=5, labelFontSize=13)),
        tooltip=['Journal', 'Count']
    )
    
    bars = base.mark_bar().encode(
        color=alt.Color('Journal', legend=None, scale=alt.Scale(range=palette), sort=alt.EncodingSortField(field="Count", order="descending"))
    )
    
    text = base.mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontSize=13 
    ).encode(
        text='Count'
    )
    
    chart = (bars + text).properties(
        title = alt.TitleParams('Publications by Journal',
                                anchor = 'start',
                                fontSize=16),
        width=PLOT_WIDTH - 100, 
        height=PLOT_HEIGHT)
    
    # Save using standard save method as requested
    chart.save(JOURNAL_OUTPUT)
    print(f"\tSaved {JOURNAL_OUTPUT}")

def generate_keyword_network(bib_data):
    print("Generating Keyword Network")
    keywords = []
    
    # Extract keywords
    for entry in bib_data.entries.values():
        if 'keywords' in entry.fields:
            # Split by comma or semicolon, normalize
            entry_keywords = entry.fields['keywords'].replace(';', ',').split(',')
            cleaned = [k.strip().lower() for k in entry_keywords if k.strip()]
            keywords.extend(cleaned)
            
    if not keywords:
        print("No keywords found.")
        return

    # Deduplicate but keep counts
    unique_keywords = list(set(keywords))
    keyword_counts = pd.Series(keywords).value_counts()
    
        
    # Display names to lowercase
    display_names = {k: k.lower() for k in unique_keywords}

    # Semantic Embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(unique_keywords)
    
    # Similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Build Graph
    G = nx.Graph()
    
    # 1. Add Keyword Nodes
    for k in unique_keywords:
        G.add_node(k, count=keyword_counts[k], label=display_names[k], node_type='keyword')
    
    # --- User-Defined Hub-and-Spoke Topology ---
    clusters = {
        'Recreation Management': [
            'visitation', 'visitor use monitoring', 'park and protected area management', 
            'park management', 'protected areas', 'urban parks', 'visitor management', 'visitor demographics'
        ],
        'Recreation Ecology': ['spatial use', 'trail impact', 'recreation ecology'],
        'Data': ['gis', 'drones', 'mobile device data', 'big data', 'phone data']
    }
    
    existing_nodes = set(unique_keywords)
    hub_nodes = []

    # 2. Create Hub Nodes and Connect Spokes
    for cluster_name, topics in clusters.items():
        # Check if this cluster has any representation
        valid_topics = [t for t in topics if t in existing_nodes]
        
        if valid_topics:
            # Add Hub Node
            G.add_node(cluster_name, count=5, label=cluster_name, node_type='hub') # Artificial count for size
            hub_nodes.append(cluster_name)
            
            # Connect spokes
            for topic in valid_topics:
                G.add_edge(topic, cluster_name, weight=2.0)

    # 3. Inter-cluster connections (Hub <-> Hub)
   
    if 'Recreation Management' in hub_nodes and 'Recreation Ecology' in hub_nodes:
        G.add_edge('Recreation Management', 'Recreation Ecology', weight=1.0)
        
    if 'Recreation Management' in hub_nodes and 'Data' in hub_nodes:
        G.add_edge('Recreation Management', 'Data', weight=1.0)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.25, iterations=100) # Slightly looser k for hubs
    
    # Edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#00a896'), # Teal edges
        hoverinfo='none',
        mode='lines')

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_type = G.nodes[node].get('node_type', 'keyword')
        
        if node_type == 'hub':
            name = node  # The label is the node name
            count = 0 # Dummy
            node_text.append(f"<b>{name}</b> (Cluster)")
            node_size.append(30) # Hubs are big
            node_color.append('#fc8d2c') # Orange Hubs
        else:
            count = G.nodes[node]['count']
            name = G.nodes[node]['label']
            node_text.append(f"{name}<br>Count: {count}")
            node_size.append(15 + count * 5) 
            node_color.append('#01648f') # Blue Keywords

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False, 
            opacity=1.0,
            color=node_color,
            size=node_size,
            line_width=1,
            line_color='black')) # Dark border for visibility
    node_trace.text = node_text
    
    annotations = []
    for hub in hub_nodes:
        cx, cy = pos[hub]
        
        # Wrap Hub Label
        label_text = hub.replace(" ", "<br>")
        
        # Add halo effect using text-shadow
        halo_style = "text-shadow: 2px 0 #fff, -2px 0 #fff, 0 2px #fff, 0 -2px #fff, 1px 1px #fff, -1px -1px #fff, 1px -1px #fff, -1px 1px #fff;"
        formatted_text = f"<span style='{halo_style}'><b>{label_text}</b></span>"
        
        annotations.append(dict(
            x=cx,
            y=cy,
            xref="x",
            yref="y",
            text=formatted_text,
            showarrow=False,
            font=dict(color="black", size=16, family="Arial"),
            bgcolor="rgba(0,0,0,0)", # Transparent background
        ))

    # Figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text='Research Topics',
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    width=PLOT_WIDTH,
                    height=PLOT_HEIGHT,
                    hovermode='closest',
                    margin=dict(b=20,l=20,r=20,t=40),
                    annotations=annotations,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Write HTML with Liquid raw tags to prevent Jekyll errors
    html_content = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})
    with open(KEYWORD_OUTPUT, 'w') as f:
        f.write(html_content)
        
    print(f"\tSaved {KEYWORD_OUTPUT}")

def scholar_citations():
    today = datetime.now().strftime('%-d %b %Y')
    print("Generating Scholar Citations")

    def parsel_scrape_author_cited_by_graph():
        time.sleep(random.uniform(5, 10)) # Wait between 5 and 10 seconds
        params = {'user': 'yWVPrskAAAAJ','hl': 'en'} 

        USER_AGENT_LIST = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0']

        headers = {'User-Agent': random.choice(USER_AGENT_LIST)}
        html = requests.get('https://scholar.google.com/citations', params=params, headers=headers, timeout=50)
        selector = Selector(text=html.text)

        data = {'cited_by': [],
                'graph': []}

        since_year = selector.css('.gsc_rsb_sth~ .gsc_rsb_sth+ .gsc_rsb_sth::text').get()

        for cited_by_public_access in selector.css('.gsc_rsb'):
            data['cited_by'].append({
                'citations_all': cited_by_public_access.css('tr:nth-child(1) .gsc_rsb_sc1+ .gsc_rsb_std::text').get(),
                f'citations_since_{since_year}': cited_by_public_access.css('tr:nth-child(1) .gsc_rsb_std+ .gsc_rsb_std::text').get(),
                'h_index_all': cited_by_public_access.css('tr:nth-child(2) .gsc_rsb_sc1+ .gsc_rsb_std::text').get(),
                f'h_index_since_{since_year}': cited_by_public_access.css('tr:nth-child(2) .gsc_rsb_std+ .gsc_rsb_std::text').get(),
                'i10_index_all': cited_by_public_access.css('tr~ tr+ tr .gsc_rsb_sc1+ .gsc_rsb_std::text').get(),
                f'i10_index_since_{since_year}': cited_by_public_access.css('tr~ tr+ tr .gsc_rsb_std+ .gsc_rsb_std::text').get(),
                'articles_num': cited_by_public_access.css('.gsc_rsb_m_a:nth-child(1) span::text').get().split(' ')[0],
                'articles_link': f"https://scholar.google.com{cited_by_public_access.css('#gsc_lwp_mndt_lnk::attr(href)').get()}"
            })

        for graph_year, graph_yaer_value in zip(selector.css('.gsc_g_t::text'), selector.css('.gsc_g_al::text')):
            data['graph'].append({
                'year': graph_year.get(),
                'value': int(graph_yaer_value.get())
            })


        return data

    data = parsel_scrape_author_cited_by_graph()


    metrics = pd.Series({'citations':int(data['cited_by'][0]['citations_all']),
                        'i10_index':int(data['cited_by'][0]['i10_index_all']),
                        'h_index':int(data['cited_by'][0]['h_index_all'])},
                        name='Metric').to_frame().reset_index()

    citations = pd.DataFrame(columns=['Year','Citations'])
    for y in range(len(data['graph'])):
        citations.loc[y,'Year'] = int(data['graph'][y]['year'])
        citations.loc[y,'Citations'] = int(data['graph'][y]['value'])

    alt.theme.enable('urbaninstitute')

    chart = alt.Chart(citations).properties(width=500,
                                            height=200,
                                            title = alt.TitleParams(
                                                [f'Data from Google Scholar | Updated: {today}'],
                                                baseline = 'bottom',
                                                orient='bottom',
                                                anchor = 'end',
                                                fontWeight='normal',
                                                fontSize=12,
                                                dy=20,
                                                dx=20)
                                ).encode(alt.X('Year:O', 
                                            title=None,
                                            axis=alt.Axis(labelAngle=0,
                                                            labelFontSize=16)),
                                        alt.Y('Citations:Q', title=None,
                                            axis=alt.Axis(format='d',
                                                            domain=False,
                                                            labelFontSize=13,
                                                            tickMinStep=2,
                                                            tickCount=10,
                                                            ticks=True)))


    area = chart.mark_area(line={'color':'dodgerblue'},
                        color=alt.Gradient(gradient='linear',
                            stops=[alt.GradientStop(color='white', offset=0),
                                    alt.GradientStop(color='dodgerblue', offset=1)],
                            x1=1, x2=1, y1=1, y2=0))

    points = chart.mark_point(filled=True,
                            size=125,
                            color='dodgerblue',
                            stroke='white',
                            strokeWidth=2)

    text_layer = chart.mark_text(align='center',
                                baseline='bottom',
                                dy=-8,
                                size=15,
                                color='black'
                                ).encode(text=alt.Text('Citations:Q', format='d'))

    main_chart = area + points + text_layer

    table = alt.Chart(metrics.reset_index()
                    ).mark_text(align='left',
                                fontSize=16,
                                font='Arial',
                                color='black'
                    ).encode(y=alt.Y('index:O', axis=None),
                            text = 'text_combined:N',
                            order= alt.Order('index')
                    ).transform_window(row_number='row_number()'
                    ).transform_calculate(
                        padded_index="{'citations': 'Citations ', 'i10_index': 'i10-index', 'h_index': 'h-index   '}[datum.index]",
                        text_combined="datum.padded_index + ': ' + datum.Metric"
                    ).properties(
                        title=alt.TitleParams('Publication Metrics', anchor='start', fontSize=18))

    final_plot = main_chart | table

    final_plot = final_plot.properties(
        title=alt.TitleParams('Noah Creany Citations', anchor='start'))

    final_plot.save('_includes/chart.html')
    print("\tSaved chart.html")


def main():
    bib_data = load_bibliography(BIB_FILE)
    generate_journal_plot(bib_data)
    # generate_keyword_network(bib_data)
    scholar_citations()


if __name__ == "__main__":
    main()