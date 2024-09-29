
    from graphviz import Graph
    gz=Graph("Bus system 12",'comment',None,None,'png',None,"UTF-8",
    {'bgcolor':'white','rankdir':'TB','splines':'ortho'},
    {'color':'black','fontcolor':'black','fontsize':'12','shape':'box','height':'0.05','length':'2','style':'filled'})
    #node description
gz.node('1','',{'color':'black','fontcolor':'black','xlabel':'bus 1'})
gz.node('2','',{'color':'black','fontcolor':'black','xlabel':'bus 2'})
gz.node('3','',{'color':'black','fontcolor':'black','xlabel':'bus 3'})
gz.node('4','',{'color':'black','fontcolor':'black','xlabel':'bus 4'})
gz.node('5','',{'color':'black','fontcolor':'black','xlabel':'bus 5'})
gz.node('6','',{'color':'black','fontcolor':'black','xlabel':'bus 6'})
gz.node('7','',{'color':'black','fontcolor':'black','xlabel':'bus 7'})
gz.node('8','',{'color':'black','fontcolor':'black','xlabel':'bus 8'})
gz.node('9','',{'color':'black','fontcolor':'black','xlabel':'bus 9'})
gz.node('10','',{'color':'black','fontcolor':'black','xlabel':'bus 10'})
gz.node('11','',{'color':'black','fontcolor':'black','xlabel':'bus 11'})
gz.node('12','',{'color':'black','fontcolor':'black','xlabel':'bus 12'})

#edge description
gz.edge('1','4','',{'color':'black'})
gz.edge('2','5','',{'color':'black'})
gz.edge('3','6','',{'color':'black'})
gz.edge('4','10','',{'color':'black'})
gz.edge('5','12','',{'color':'black'})
gz.edge('6','12','',{'color':'black'})
gz.edge('7','1','',{'color':'black'})
gz.edge('8','9','',{'color':'black'})
gz.edge('9','10','',{'color':'black'})
gz.edge('10','11','',{'color':'black'})
gz.edge('11','4','',{'color':'black'})
gz.edge('12','9','',{'color':'black'})
gz.edge('5','7','',{'color':'black'})
gz.edge('6','4','',{'color':'black'})
gz.edge('3','9','',{'color':'black'})


    print(gz.source)
    gz.render(filename = "bus_system_12_20240920_192433",directory="Output/Distribution Grid/Image/")
    