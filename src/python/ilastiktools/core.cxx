/************************************************************************/
/*                                                                      */
/*                 Copyright 2011 by Ullrich Koethe                     */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#define PY_ARRAY_UNIQUE_SYMBOL vigranumpyilastiktools_PyArray_API
//#define NO_IMPORT_ARRAY

#include <utility>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <assert>

// Include this first to avoid name conflicts for boost::tie,
// similar to issue described in vigra#237
#include <boost/tuple/tuple.hpp>
#include <boost/functional/hash.hpp>

/*vigra*/
#include <ilastiktools/carving.hxx>


/*vigra python */
#include <ilastiktools/python_vigra_converter.hxx>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace vigra;


bool checkSerializationValidity(const MultiArrayView<1, UInt32>& serialization)
{
    if(serialization.size() < 4)
        throw std::invalid_argument("Need at least 4 values for deserialization");

    auto nodeNum = serialization(0);
    auto edgeNum = serialization(1);
    auto maxNodeId = serialization(2);
    auto maxEdgeId = serialization(3);

    std::cout << "Found graph that should have " << nodeNum << " nodes and " << edgeNum << " edges, with maxIds: " << maxNodeId << ", " << maxEdgeId << std::endl;
    std::cout << "checking node degrees" << std::endl;

    size_t sumOfNodeDegrees = 0;
    const size_t offset = 4 + 2 * edgeNum + 1;
    for(size_t i = 0; i < nodeNum; i++){
        if(offset + 2*i + 2*sumOfNodeDegrees > serialization.size())
            throw std::runtime_error("tried to access node definitions outside of the provided array for deserialization");
        size_t nodeDegree = serialization(offset + 2*i + 2*sumOfNodeDegrees);
        sumOfNodeDegrees += nodeDegree;
    }

    std::cout << "Total sum of node degrees is " << sumOfNodeDegrees << std::endl;


    if(serialization.size() < 4 + 2* (nodeNum + edgeNum) + 2*sumOfNodeDegrees)
    {
        std::cerr << "Expected size of " << 4 + 2* (nodeNum + edgeNum) + 2*sumOfNodeDegrees << " but got " << serialization.size() << std::endl;
        throw std::invalid_argument("Array for deserialization does not contain enough values!");
    }

    return true;
}

template<unsigned int DIM, class LABEL_TYPE>
void pyAssignLabels(
    GridRag<DIM, LABEL_TYPE> & graph,
    const py::array_t<LABEL_TYPE, py::array::f_style | py::array::forcecast>& pylabels
){
    auto labels = numpy_to_vigra<DIM, LABEL_TYPE>(pylabels);
    graph.assignLabels(labels);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyAssignLabelsFromSerialization(
    GridRag<DIM, LABEL_TYPE> & graph,
    const py::array_t<LABEL_TYPE, py::array::f_style | py::array::forcecast>& pylabels,
    const py::array_t<UInt32, py::array::f_style | py::array::forcecast>& pyserialization
){
    auto labels = numpy_to_vigra<DIM, LABEL_TYPE>(pylabels);
    auto serialization = numpy_to_vigra<1, UInt32>(pyserialization);
    graph.assignLabelsFromSerialization(labels, serialization);
}


template<unsigned int DIM, class LABEL_TYPE, class FEATURES_IN>
py::array_t<typename NumericTraits<FEATURES_IN>::RealPromote, py::array::f_style | py::array::forcecast> pyAccumulateEdgeFeatures(
    GridRag<DIM, LABEL_TYPE> & graph,
    const py::array_t<FEATURES_IN, py::array::f_style | py::array::forcecast>& pyfeatures,
    py::array_t<typename NumericTraits<FEATURES_IN>::RealPromote, py::array::f_style | py::array::forcecast>& pyout
){
    auto featuresIn = numpy_to_vigra<DIM, FEATURES_IN>(pyfeatures);
    auto out = numpy_to_vigra<1, typename NumericTraits<FEATURES_IN>::RealPromote>(pyout);
    if(out.size() == 0)
        out = MultiArray<1, typename NumericTraits<FEATURES_IN>::RealPromote>(graph.edgeNum());
    graph.accumulateEdgeFeatures(featuresIn, out);
    return vigra_to_numpy<1, typename NumericTraits<FEATURES_IN>::RealPromote>(out);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyPreprocessing(
    GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<LABEL_TYPE, py::array::f_style | py::array::forcecast>& pylabels,
    const py::array_t<float, py::array::f_style | py::array::forcecast>& pyweights
){
    auto labels = numpy_to_vigra<DIM, LABEL_TYPE>(pylabels);
    auto weightArray = numpy_to_vigra<DIM, float>(pyweights);
    gridSegmentor.preprocessing(labels, weightArray);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyPreprocessingFromSerialization(
    GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<LABEL_TYPE, py::array::f_style | py::array::forcecast>& pyLabels,
    const py::array_t<LABEL_TYPE, py::array::f_style | py::array::forcecast>& pySerialization,
    const py::array_t<float, py::array::f_style | py::array::forcecast>& pyEdgeWeights,
    const py::array_t<UInt8, py::array::f_style | py::array::forcecast>& pyNodeSeeds,
    const py::array_t<UInt8, py::array::f_style | py::array::forcecast>& pyResultSegmentation
){
    auto labels = numpy_to_vigra<DIM, LABEL_TYPE>(pyLabels);
    auto serialization = numpy_to_vigra<1, LABEL_TYPE>(pySerialization);
    auto edgeWeights = numpy_to_vigra<1, float>(pyEdgeWeights);
    auto nodeSeeds = numpy_to_vigra<1, UInt8>(pyNodeSeeds);
    auto resultSegmentation = numpy_to_vigra<1, UInt8>(pyResultSegmentation);

    assert(checkSerializationValidity(serialization));

    gridSegmentor.preprocessingFromSerialization(labels, serialization,
                                                 edgeWeights, nodeSeeds,
                                                 resultSegmentation);
}


template<unsigned int DIM, class LABEL_TYPE>
void pyAddLabels(
    GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<UInt8, py::array::f_style | py::array::forcecast>& pyBrushStroke,
    const py::array_t<MultiArrayIndex, py::array::f_style | py::array::forcecast> pyRoiBegin,
    const py::array_t<MultiArrayIndex, py::array::f_style | py::array::forcecast> pyRoiEnd,
    const UInt8 maxValidLabel
){
    auto brushStroke = numpy_to_vigra<DIM, UInt8>(pyBrushStroke);
    auto roiBegin = numpy_to_tiny_vector<DIM, MultiArrayIndex>(pyRoiBegin);
    auto roiEnd = numpy_to_tiny_vector<DIM, MultiArrayIndex>(pyRoiEnd);
    gridSegmentor.addLabels(brushStroke, roiBegin, roiEnd, maxValidLabel);
}

template<unsigned int DIM, class LABEL_TYPE>
py::array_t<UInt8, py::array::f_style | py::array::forcecast> pyGetSegmentation(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<MultiArrayIndex, py::array::f_style | py::array::forcecast> pyRoiBegin,
    const py::array_t<MultiArrayIndex, py::array::f_style | py::array::forcecast> pyRoiEnd,
    py::array_t<UInt8, py::array::f_style | py::array::forcecast>* pySegmentation
){
    auto roiBegin = numpy_to_tiny_vector<DIM, MultiArrayIndex>(pyRoiBegin);
    auto roiEnd = numpy_to_tiny_vector<DIM, MultiArrayIndex>(pyRoiEnd);

    MultiArrayView<DIM, UInt8> segmentation;
    if(pySegmentation == nullptr)
         segmentation = MultiArray<DIM, UInt8>(roiEnd - roiBegin);
    else
        segmentation = numpy_to_vigra<DIM, UInt8>(*pySegmentation);

    if(segmentation.shape() != roiEnd - roiBegin)
    {
        throw std::invalid_argument("Dimensions must match!");
    }

    {
        pybind11::gil_scoped_release release;
        gridSegmentor.getSegmentation(roiBegin, roiEnd, segmentation);
    }
   
    return vigra_to_numpy<DIM, UInt8>(segmentation);
}


template<unsigned int DIM, class LABEL_TYPE>
py::array_t<UInt8, py::array::f_style | py::array::forcecast> pyGetSuperVoxelSeg(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor
){
    auto segmentation = MultiArray<1, UInt8>(gridSegmentor.maxNodeId()+1);
    
    {
        pybind11::gil_scoped_release release;
        gridSegmentor.getSuperVoxelSeg(segmentation);
    }
   
    return vigra_to_numpy<1, UInt8>(segmentation);
}

template<unsigned int DIM, class LABEL_TYPE>
py::array_t<UInt8, py::array::f_style | py::array::forcecast> pyGetSuperVoxelSeeds(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor
){
    auto seeds = MultiArray<1, UInt8>(gridSegmentor.maxNodeId()+1);

    {
        pybind11::gil_scoped_release release;
        gridSegmentor.getSuperVoxelSeeds(seeds);
    }
   
    return vigra_to_numpy<1, UInt8>(seeds);
}


template<unsigned int DIM, class LABEL_TYPE>
py::array_t<UInt32, py::array::f_style | py::array::forcecast> pySerializeGraph(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    py::array_t<UInt32, py::array::f_style | py::array::forcecast>* pySerialization 
){
    MultiArrayView<1, UInt32> serialization;
    if(pySerialization == nullptr)
         serialization = MultiArray<1, UInt32>(gridSegmentor.graph().serializationSize());
    else
        serialization = numpy_to_vigra<1, UInt32>(*pySerialization);

    if(serialization.size() != gridSegmentor.graph().serializationSize())
    {
        throw std::invalid_argument("Dimensions must match!");
    }

    gridSegmentor.graph().serialize(serialization.begin());
    return vigra_to_numpy<1, UInt32>(serialization);
}

template<unsigned int DIM, class LABEL_TYPE>
void pyDeserializeGraph(
    GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<UInt32, py::array::f_style | py::array::forcecast>& pySerialization
){
    auto serialization = numpy_to_vigra<1, UInt32>(pySerialization);

    assert(checkSerializationValidity(serialization));

    gridSegmentor.graph().clear();
    gridSegmentor.graph().deserialize(serialization.begin(),serialization.end());
}


template<unsigned int DIM, class LABEL_TYPE>
py::array_t<float, py::array::f_style | py::array::forcecast> pyEdgeWeights(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor
){
    auto out = MultiArray<1, float>(gridSegmentor.edgeNum());
    
    out = gridSegmentor.edgeWeights();
    return vigra_to_numpy<1, float>(out);
}

template<unsigned int DIM, class LABEL_TYPE>
py::array_t<UInt8, py::array::f_style | py::array::forcecast> pyNodeSeeds(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor
){
    auto out = MultiArray<1, UInt8>(gridSegmentor.maxNodeId()+1);
    
    out = gridSegmentor.nodeSeeds();
    return vigra_to_numpy<1, UInt8>(out);
}

template<unsigned int DIM, class LABEL_TYPE>
py::array_t<UInt8, py::array::f_style | py::array::forcecast> pyGetResultSegmentation(
    const GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor
){
    auto out = MultiArray<1, UInt8>(gridSegmentor.maxNodeId()+1);

    out = gridSegmentor.resultSegmentation();
    return vigra_to_numpy<1, UInt8>(out);
}


template<unsigned int DIM, class LABEL_TYPE>
void pySetSeeds(
    GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<Int64, py::array::f_style | py::array::forcecast>& pyFgSeeds,
    const py::array_t<Int64, py::array::f_style | py::array::forcecast>& pyBgSeeds
){
    auto fgSeeds = numpy_to_vigra<2, Int64>(pyFgSeeds);
    auto bgSeeds = numpy_to_vigra<2, Int64>(pyBgSeeds);
    gridSegmentor.setSeeds(fgSeeds, bgSeeds);
}

template<unsigned int DIM, class LABEL_TYPE>
void pySetResulFgObj(
    GridSegmentor<DIM , LABEL_TYPE, float> & gridSegmentor,
    const py::array_t<Int64, py::array::f_style | py::array::forcecast>& pyFgNodes
){
    auto fgNodes = numpy_to_vigra<1, Int64>(pyFgNodes);
    gridSegmentor.setResulFgObj(fgNodes);
}

template<unsigned int DIM, class LABEL_TYPE>
void defineGridRag(py::module& module, const std::string & clsName){


    typedef GridRag<DIM, LABEL_TYPE> Graph;

    py::class_<Graph>(module, clsName.c_str())
        .def(py::init< >())
        .def("assignLabels",&pyAssignLabels<DIM, LABEL_TYPE>)
        .def("accumulateEdgeFeatures", &pyAccumulateEdgeFeatures<DIM, LABEL_TYPE, float>, 
            py::arg("features"), 
            py::arg("out").none(false))
    ;
}




template<unsigned int DIM, class LABEL_TYPE>
void defineGridSegmentor(py::module& module, const std::string & clsName){


    typedef GridSegmentor<DIM, LABEL_TYPE, float> Segmentor;

    py::class_<Segmentor>(module, clsName.c_str())
        .def(py::init< >())
        .def("preprocessing", &pyPreprocessing<DIM, LABEL_TYPE>, py::arg("labels"), py::arg("weightArray"))
        .def("preprocessingFromSerialization", 
             &pyPreprocessingFromSerialization<DIM, LABEL_TYPE>,
                py::arg("labels"),
                py::arg("serialization"),
                py::arg("edgeWeights"),
                py::arg("nodeSeeds"),
                py::arg("resultSegmentation")
        )
        .def("addSeeds", 
             & pyAddLabels<DIM, LABEL_TYPE>,
                py::arg("brushStroke"),
                py::arg("roiBegin"),
                py::arg("roiEnd"),
                py::arg("maxValidLabel")
        )
        .def("setSeeds", 
             & pySetSeeds<DIM, LABEL_TYPE>,
                py::arg("fgSeeds"),
                py::arg("bgSeeds")
        )
        .def("setResulFgObj", 
             & pySetResulFgObj<DIM, LABEL_TYPE>,
                py::arg("fgNodes")
        )

        .def("getSegmentation", 
             & pyGetSegmentation<DIM, LABEL_TYPE>,
                py::arg("roiBegin"),
                py::arg("roiEnd"),
                py::arg("out").none(false)
        )
        .def("getSegmentation", 
            [](const GridSegmentor<DIM , LABEL_TYPE, float> &g,
               const py::array_t<MultiArrayIndex, py::array::f_style | py::array::forcecast> roiBegin,
               const py::array_t<MultiArrayIndex, py::array::f_style | py::array::forcecast> roiEnd)
               ->py::array_t<UInt8, py::array::f_style | py::array::forcecast>{ 
                    return pyGetSegmentation<DIM, LABEL_TYPE>(g, roiBegin, roiEnd, nullptr);
            },
            py::arg("roiBegin"),
            py::arg("roiEnd")
        )
        .def("nodeNum",&Segmentor::nodeNum)
        .def("edgeNum",&Segmentor::edgeNum)
        .def("maxNodeId",&Segmentor::maxNodeId)
        .def("maxEdgeId",&Segmentor::maxEdgeId)
        .def("run",&Segmentor::run)
        .def("clearSeeds",&Segmentor::clearSeeds)
        .def("clearSegmentation",&Segmentor::clearSegmentation)
        .def("serializeGraph", &pySerializeGraph<DIM, LABEL_TYPE>, py::arg("out").none(false))
        .def("serializeGraph", [](const GridSegmentor<DIM , LABEL_TYPE, float> &g)->py::array_t<UInt32, py::array::f_style | py::array::forcecast>{ return pySerializeGraph<DIM, LABEL_TYPE>(g, nullptr);})
        .def("deserializeGraph", &pyDeserializeGraph<DIM, LABEL_TYPE>, py::arg("serialization"))
        .def("getEdgeWeights",pyEdgeWeights<DIM, LABEL_TYPE>)
        .def("getNodeSeeds",pyNodeSeeds<DIM, LABEL_TYPE>)
        .def("getResultSegmentation",pyGetResultSegmentation<DIM, LABEL_TYPE>)
        .def("getSuperVoxelSeg",pyGetSuperVoxelSeg<DIM, LABEL_TYPE>)
        .def("getSuperVoxelSeeds",pyGetSuperVoxelSeeds<DIM, LABEL_TYPE>)
    ;
}

// Define lookup type:
// (u,v) -> [(x,y), (x,y),...]
typedef std::pair<UInt32, UInt32> edge_id_t;
typedef std::unordered_map<edge_id_t,
                           std::vector<Shape2>,
                           boost::hash<edge_id_t> > edge_coord_lookup_t;

typedef std::pair<edge_coord_lookup_t, edge_coord_lookup_t> edge_coord_lookup_pair_t;

//*****************************************************************************
//* Given a 2D label image, find all the coordinates just *before* a label
//* transition in both the x and y directions.
//*
//* Returns a pair of mappings of edge pairs (u,v) to coordinate lists; one for
//* edges in the horizontal direction, and another for the vertical direction.
//*
//* For every edge pair (u,v): u < v.
//* No ordering guarantee is made for the coordinate lists (but in practice,
//* they will be in scan order).
//*
//* As a convenience, every key in the horizontal lookup is guaranteed to exist
//* in the vertical lookup (and vice-versa), even if that key's coordinate list
//* is empty.
//*
//*****************************************************************************
edge_coord_lookup_pair_t edgeCoords2D( const MultiArrayView<2, UInt32>& src )
{
    MultiArrayIndex x_dim = src.shape(0);
    MultiArrayIndex y_dim = src.shape(1);

    edge_coord_lookup_t horizontal_edge_coords;
    edge_coord_lookup_t vertical_edge_coords;

    for (MultiArrayIndex y = 0; y < y_dim; ++y)
    {
        for (MultiArrayIndex x = 0; x < x_dim; ++x)
        {
            // Lambda to append to a lookup
            auto append_to_lookup =
            [&](edge_coord_lookup_t & lookup, MultiArrayIndex x1, MultiArrayIndex y1)
            {
                auto u = src(x,y);
                auto v = src(x1,y1);
                edge_id_t edge_id = (u < v) ? std::make_pair(u,v) : std::make_pair(v,u);

                auto iter = lookup.find(edge_id);
                if ( iter == lookup.end() )
                {
                    // Edge not yet seen. Create a new coord vector
                    auto coord_list = std::vector<Shape2>();
                    coord_list.push_back(Shape2(x,y));
                    lookup[edge_id] = coord_list;
                }
                else
                {
                    // Append to coord vector
                    auto & coord_list = iter->second;
                    coord_list.push_back(Shape2(x,y));
                }
            };

            // Check to the right
            if (x < x_dim-1 && src(x,y) != src(x+1,y))
            {
                append_to_lookup(horizontal_edge_coords, x+1, y);
            }

            // Check below
            if (y < y_dim-1 && src(x,y) != src(x,y+1))
            {
                append_to_lookup(vertical_edge_coords, x, y+1);
            }
        }
    }

    // Convenience feature:
    // Ensure that every pair (u,v) in horizontal_edge_coords appears in vertical_edge_coords
    // and vice-versa, even if it maps to an empty coordinate list in one of them.
    auto fill_missing_keys =
    [](edge_coord_lookup_t const & from, edge_coord_lookup_t & to)
    {
        for ( auto const & k_v : from )
        {
            if ( to.find(k_v.first) == to.end() )
            {
                to[k_v.first] = std::vector<Shape2>();
            }
        }
    };
    fill_missing_keys(horizontal_edge_coords, vertical_edge_coords);
    fill_missing_keys(vertical_edge_coords, horizontal_edge_coords);

    return std::make_pair(horizontal_edge_coords, vertical_edge_coords);
}

//*****************************************************************************
//* Convert an edge_coord_lookup_t to a corresponding python structure.
//* Result is a dict of { tuple : list-of-tuple }, like this:
//*   { (u,v) : [(x,y), (x,y), (x,y), ...] }
//*****************************************************************************
py::dict edgeCoordLookupToPython( edge_coord_lookup_t const & edge_coord_lookup )
{
    py::dict pylookup;
    for ( auto & edge_and_coords : edge_coord_lookup )
    {
        edge_id_t const & edge_id = edge_and_coords.first;
        std::vector<Shape2> const & coords = edge_and_coords.second;

        py::list pycoords;
        for ( auto coord : coords )
        {
            pycoords.append(py::make_tuple(coord[0], coord[1]));
        }
        pylookup[py::make_tuple(edge_id.first, edge_id.second)] = pycoords;
    }
    return pylookup;
}

//*****************************************************************************
//* Python function for edgeCoords2D().
//* See edgeCoords2D() documentation for details.
//*****************************************************************************
py::tuple pythonEdgeCoords2D(const py::array_t<UInt32, py::array::f_style | py::array::forcecast>& pySrc)
{
    auto src = numpy_to_vigra<2, UInt32>(pySrc);
    edge_coord_lookup_pair_t lookup_pair;
    {
        pybind11::gil_scoped_release release;
        lookup_pair = edgeCoords2D(src); // C++ move constructor should work here ...right?
    }

    py::dict pycoords_horizontal = edgeCoordLookupToPython(lookup_pair.first);
    py::dict pycoords_vertical = edgeCoordLookupToPython(lookup_pair.second);
    return py::make_tuple( pycoords_horizontal, pycoords_vertical );
}


//*****************************************************************************
//* Find all the label transitions for the given 2D label image.
//* Then return a dict-of-lists mapping each edge (u,v) to a list of line
//* segments that could be drawn on screen to represent the edge.
//*
//* Conceptually, the output looks like this:
//*
//* { (u,v) : [ ((x1,y1), (x2,y2)),
//*             ((x1,y1), (x2,y2)),
//*             ((x1,y1), (x2,y2)),
//*             ... ] }
//*
//* ...but the line segment list is returned as a NumpyArray of shape (N,2,2).
//*
//* Note: The line segments are not guaranteed to appear in any particular order.
//*
//* TODO: It would be nice if we could use PyAllowThreads here, but I'm not
//*       sure if it's allowed -- we create NumpyArrays inside the loop
//*       (of type line_segment_array_t) inside the loop.
//*
//*****************************************************************************
py::dict line_segments_for_labels( const py::array_t<UInt32, py::array::f_style | py::array::forcecast>& pyLabelImg )
{
    auto label_img = numpy_to_vigra<2, UInt32>(pyLabelImg);
    // typedef NumpyArray<3, UInt32> line_segment_array_t;
    typedef py::array_t<UInt32, py::array::f_style | py::array::forcecast> line_segment_array_t;
    typedef std::unordered_map<edge_id_t, line_segment_array_t, boost::hash<edge_id_t> > line_segment_lookup_t;

    auto lookup_pair = edgeCoords2D(label_img);
    auto const & horizontal_coord_lookup = lookup_pair.first;
    auto const & vertical_coord_lookup = lookup_pair.second;

    typedef std::unordered_set<edge_id_t, boost::hash<edge_id_t> > edge_id_set_t;
    edge_id_set_t all_edge_ids;
    for ( auto const & k_v : horizontal_coord_lookup )
    {
        all_edge_ids.insert(k_v.first);
    }
    for ( auto const & k_v : vertical_coord_lookup )
    {
        all_edge_ids.insert(k_v.first);
    }

    line_segment_lookup_t line_seg_lookup;
    for ( auto const & edge_id : all_edge_ids )
    {
        // Empty by default
        std::vector<Shape2> horizontal_edge_coords;
        std::vector<Shape2> vertical_edge_coords;

        // Overwrite if found
        auto iter_horizontal_coords = horizontal_coord_lookup.find(edge_id);
        if ( iter_horizontal_coords != horizontal_coord_lookup.end() )
        {
            horizontal_edge_coords = iter_horizontal_coords->second;
        }

        // Overwrite if found
        auto iter_vertical_coords = vertical_coord_lookup.find(edge_id);
        if ( iter_vertical_coords != vertical_coord_lookup.end() )
        {
            vertical_edge_coords = iter_vertical_coords->second;
        }

        auto num_segments = horizontal_edge_coords.size() + vertical_edge_coords.size();
        auto line_segments = MultiArray<3, UInt32>(Shape3(num_segments, 2, 2));

        // Line segments to the RIGHT of the HORIZONTAL edge coordinates
        for ( int i = 0; i < horizontal_edge_coords.size(); ++i )
        {
            line_segments(i, 0, 0) = horizontal_edge_coords[i][0] + 1;
            line_segments(i, 0, 1) = horizontal_edge_coords[i][1] + 0;
            line_segments(i, 1, 0) = horizontal_edge_coords[i][0] + 1;
            line_segments(i, 1, 1) = horizontal_edge_coords[i][1] + 1;
        }
        // Line segments BELOW the VERTICAL edge coordinates
        auto offset = horizontal_edge_coords.size();
        for ( int i = 0; i < vertical_edge_coords.size(); ++i )
        {
            line_segments(offset+i, 0, 0) = vertical_edge_coords[i][0] + 0;
            line_segments(offset+i, 0, 1) = vertical_edge_coords[i][1] + 1;
            line_segments(offset+i, 1, 0) = vertical_edge_coords[i][0] + 1;
            line_segments(offset+i, 1, 1) = vertical_edge_coords[i][1] + 1;
        }

        line_seg_lookup[edge_id] = vigra_to_numpy<3, UInt32>(line_segments);
    }

    py::dict ret;
    for ( auto const & k_v : line_seg_lookup )
    {
        auto const & edge_id = k_v.first;
        ret[py::make_tuple(edge_id.first, edge_id.second)] = k_v.second;
    }
    return ret;
}



PYBIND11_MODULE(_core, m)
{
    defineGridRag<2, vigra::UInt32>(m, "GridRag_2D_UInt32");
    defineGridSegmentor<2, vigra::UInt32>(m, "GridSegmentor_2D_UInt32");


    defineGridRag<3, vigra::UInt32>(m, "GridRag_3D_UInt32");
    defineGridSegmentor<3, vigra::UInt32>(m, "GridSegmentor_3D_UInt32");

    m.def("edgeCoords2D", &pythonEdgeCoords2D, py::arg("src"));
    m.def("line_segments_for_labels", &line_segments_for_labels, (py::arg("label_img")));
}


