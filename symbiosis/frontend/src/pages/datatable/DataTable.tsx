"use client";
import React, {useEffect, useMemo, useState} from "react";
import MUIDataTable from "mui-datatables";
import TableRow from "@mui/material/TableRow"; // Import TableRow
import TableCell from "@mui/material/TableCell";
import Chip from '@mui/material/Chip';
import {Card, CircularProgress, TextField} from '@mui/material';
import {useRouter} from 'next/router';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Checkbox from '@mui/material/Checkbox';

// Function to extract model types from the models data
const getModelTypes = (models) => {
  try {
    const modelTypes = new Set();

    if (models != null) {
      for (const model in models) {
        if (models[model].is_sfd == true) {
          modelTypes.add("Stock-and-Flow");
        } else if (models[model].is_cld == true) {
          modelTypes.add("Causal-Loop-Diagram");
        } else {
          modelTypes.add("None");
        }
      }
      return Array.from(modelTypes);
    } else {
      return ["None"];
    }
  } catch (error) {
    console.error("Error parsing models:", error);
    return ["None"];
  }
};

// Define the columns for the table
const columns = [
  {
    name: 'uuid',
    label: 'UUID',
    options: {
      filter: false,
      sort: false,
      display: false
    }
  },
  {
    name: "paper_title",
    label: "Paper Title",
    options: {
      filter: true,
      sort: true,
    },
  },
  {
    name: "authors",
    label: "Authors",
    options: {
      filter: false,
      sort: false,
    },
  },
  {
    name: "sdgs",
    label: "SDGs",
    options: {
      filter: true,
      sort: true,
      customBodyRender: (value) => {
        try {
          console.log(value);
          return value.length > 0 ? value.toString() : "None";
        } catch (error) {
          console.error("Error parsing SDGs values:", error);
          return "None";
        }
      },
    },
  },
  {
    name: "models",
    label: "Number of Models",
    options: {
      filter: true,
      sort: false,
      display: true,
      customBodyRender: (value) => {
        try {
          // Attempt to parse the value
          return value ? Object.keys(JSON.parse(value)).length : 0;
        } catch (error) {
          console.error("Error parsing models:", error);
          return 0;
        }
      },
      filterType: 'custom',
      filterOptions: {
        names: ['1', '2', '3', '4', '5+'], // Predefined filter names for the dropdown
        logic(models, filterValue) {
          const numModels = models ? Object.keys(JSON.parse(models)).length : 0;
          if (filterValue === '5+') {
            return numModels >= 5;
          } else {
            return numModels === parseInt(filterValue);
          }
        },
        display: (filterList, onChange, index, column) => (
            <FormControl sx={{m: 1, minWidth: 120}}>
              <InputLabel id="model-count-label">Model Count</InputLabel>
              <Select
                  labelId="model-count-label"
                  id="model-count-select"
                  value={filterList[index][0] || ''}
                  onChange={(event) => {
                    filterList[index][0] = event.target.value;
                    onChange(filterList[index], index, column);
                  }}
                  label="Model Count"
              >
                <MenuItem value="">All</MenuItem> {/* Add an "All" option */}
                {['1', '2', '3', '4', '5+'].map((count) => (
                    <MenuItem key={count} value={count}>{count}</MenuItem>
                ))}
              </Select>
            </FormControl>
        )
      }
    }
  },
  {
    name: "models",
    label: "Model Type",
    options: {
      filter: true,
      sort: true,
      display: true,
      customBodyRender: (value) => {
        try {
          // Attempt to parse the value
          const models = JSON.parse(value);
          return getModelTypes(models);
        } catch (error) {
          console.error("Error parsing models:", error);
          return ["None"];
        }
      },
      names: ["Stock-and-Flow", "Causal-Loop-Diagram", "None"], // Hardcoded values
      logic(models, filterValue) {
        const modelTypes = getModelTypes(models);
        return modelTypes.includes(filterValue);
      },
      display: (filterList, onChange, index, column) => (
          <FormGroup>
            {["Stock-and-Flow", "Causal-Loop-Diagram", "None"].map((value) => (
                <FormControlLabel
                    key={getModelTypes(value)}
                    control={
                      <Checkbox
                          checked={filterList[index].includes(getModelTypes(value))}
                          onChange={(event) => {
                            if (event.target.checked) {
                              filterList[index].push(getModelTypes(value));
                            } else {
                              filterList[index] = filterList[index].filter((v) => v !== getModelTypes(value));
                            }
                            onChange(filterList[index], index, column);
                          }}
                      />
                    }
                    label={getModelTypes(value)}
                />
            ))}
          </FormGroup>
      )
    },
  },
];


const DataTable = () => {
  const [data, setData] = useState([]);
  const [page, setPage] = useState(0);
  const [count, setCount] = useState(0);
  const [sortOrder, setSortOrder] = useState({});
  const [uuid, setUuid] = useState(null);
  const [overallSearch, setOverallSearch] = useState(true);
  const [uuidSearch, setUuidSearch] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // Extract uuid from query parameters
    const {uuid: queryUuid} = router.query;

    if (queryUuid) {
      setUuid(queryUuid);
      setUuidSearch(true);
      setOverallSearch(false);
    }
  }, [router.query]);

  useEffect(() => {
    if (overallSearch) {
      const fetchData = async () => {
        setIsLoading(true);
        try {
          const encodedPath = encodeURIComponent(`/all_papers_data?page=${page}&size=10`);
          const response = await fetch(`$$BACKEND_URL$$?path=${encodedPath}`);
          const data = await response.json();
          console.log(data.message);

          setData(data.message.papers);
          setCount(data.message.count);
        } catch (error) {
          console.error('Error fetching or parsing data:', error);
        } finally {
          setIsLoading(false);
        }
      };

      fetchData();
    }
  }, [overallSearch, page, sortOrder]);

  useEffect(() => {
    if (uuidSearch) {
      const fetchData = async () => {
        setIsLoading(true);
        setOverallSearch(false);
        try {
          let encodedPath;
          // if (uuid) {
          console.log('Fetching with UUID: ', uuid);
          encodedPath = encodeURIComponent(`/all_papers_data?uuid=${uuid}`);
          // } else {
          //   encodedPath = encodeURIComponent(`/all_papers_data?page=${page}&size=10`);
          // }
          const response = await fetch(`$$BACKEND_URL$$?path=${encodedPath}`);
          const data = await response.json();
          console.log(data.message);

          setData(data.message.papers);
          setPage(null);
          setCount(null);
          console.log(data);
          console.log()
          // setCount(data.message.count);
        } catch (error) {
          console.error('Error fetching or parsing data:', error);
        } finally {
          setIsLoading(false);
        }
      };
      fetchData();
    }
  }, [uuidSearch]);

  // Options for the table
  const options = {
    filterType: "checkbox", // Use checkbox for filtering
    expandableRows: true, // Enable expandable rows
    renderExpandableRow: (rowData: any, rowMeta: any) => {
      // Render the content of the expandable row
      const colSpan = rowData.length + 1;
      const models = data[rowMeta.dataIndex].models;
      const parsedModels = models ? JSON.parse(models) : {};
      const modelKeys = Object.keys(parsedModels);
      const abstract = data[rowMeta.dataIndex].abstract;

      return (
          <TableRow>
            <TableCell colSpan={colSpan}>
              {modelKeys.map((key) => (
                  <Chip
                      key={key}
                      label={key}
                      onClick={() => router.push(`/explore/model?uuid=${data[rowMeta.dataIndex].uuid}&modelKey=${key}`)}
                      sx={{marginRight: 1}} // Add some spacing between chips
                  />
              ))}
              <Card sx={{minWidth: 200}}>
                <CardContent>
                  <Typography gutterBottom
                              sx={{color: 'text.secondary', fontSize: 14}}>
                    Abstract: {abstract}
                  </Typography>
                </CardContent>
              </Card>
            </TableCell>
          </TableRow>
      );
    },
    pagination: true,
    print: false,
    download: false,
    count: count,
    page: page,
    rowsPerPage: 10,
    rowsPerPageOptions: [10, 20, 50],
    onChangePage: (newPage) => setPage(newPage),
    serverSide: true,
    onTableChange: (action, tableState) => {
      console.log(action, tableState);

      switch (action) {
        case 'changePage':
          setPage(tableState.page);
          break;
        case 'sort':
          setSortOrder(tableState.sortOrder);
          break;
        default:
          console.log('action not handled.');
          console.log(action);
      }
    },
  };

  return (
      <div>
        {isLoading ? (
            <CircularProgress />
        ) : (
            <MUIDataTable
                title={"Systems Thinking Research Papers"}
                data={data}
                columns={columns}
                options={options}
            />
        )}
      </div>
  );
};

export default DataTable;