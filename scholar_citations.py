from parsel import Selector
import requests, os, json
import pandas as pd
import altair as alt

def parsel_scrape_author_cited_by_graph():
    params = {'user': 'yWVPrskAAAAJ',
              'hl': 'en'} 

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }

    html = requests.get('https://scholar.google.com/citations', params=params, headers=headers, timeout=100)
    selector = Selector(text=html.text)

    data = {
        'cited_by': [],
        'graph': []
    }

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


chart = alt.Chart(citations).properties(width=600,
                                        height=350,
                                        title = alt.TitleParams(
                                            ['Data from Google Scholar'],
                                            baseline = 'bottom',
                                            orient='bottom',
                                            anchor = 'end',
                                            fontWeight='normal',
                                            fontSize=12,
                                            dy=20,
                                            dx=20)
                            ).encode(alt.X('Year:O', 
                                           title='Year',
                                           axis=alt.Axis(labelAngle=0,
                                                         labelFontSize=14,
                                                         titleFontSize=16)),
                                     alt.Y('Citations:Q', title=None,
                                           axis=alt.Axis(format='d',
                                                         domain=False,
                                                         tickMinStep=2,
                                                         tickCount=10,
                                                         ticks=True)))


area = chart.mark_area(line={'color':'dodgerblue'},
                       color=alt.Gradient(
                           gradient='linear',
                           stops=[alt.GradientStop(color='white', offset=0),
                                  alt.GradientStop(color='dodgerblue', offset=1)],
                           x1=1, x2=1, y1=1, y2=0))

points = chart.mark_point(filled=True,
                         size=100,
                         color='dodgerblue',
                         stroke='white',
                         strokeWidth=2)

text_layer = chart.mark_text(align='center',
                            baseline='bottom',
                            dy=-8,
                            size=14,
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
