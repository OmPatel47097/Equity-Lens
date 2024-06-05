"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Customized,
  Rectangle,
  Bar,
  ComposedChart,
  Label,
} from "recharts";
import CustomTooltip from "./CustomTooltip";

const StockChart = () => {
  const [symbol, setSymbol] = useState("");
  const [period, setPeriod] = useState("1mo");
  const [history, setHistory] = useState([]);
  const [yRange, setYRange] = useState([0, 100]);
  const [isCandleStick, setIsCandleStick] = useState(false);
  const [isLow, setIsLow] = useState(false);
  const [isHigh, setIsHigh] = useState(false);
  const [symbols, setSymbols] = useState([]);

  const baseUrl = "http://127.0.0.1:5000/api";
  const periods = [
    // "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "ytd",
    "max",
  ];

  useEffect(() => {
    async function fetchSymbols() {
      const response = await fetch(`${baseUrl}/symbols`);
      const output = await response.json();
      const data = output.data;
      setSymbols(data);
      setSymbol(data[0].Symbol);
    }

    fetchSymbols();
  }, []);

  useEffect(() => {
    async function fetchData() {
      if (symbol) {
        const response = await fetch(`${baseUrl}/stock/${symbol}/${period}`);
        const output = await response.json();
        var data = output.data;
        data = data.map((item) => ({
          ...item,
          Date: new Date(item.Date),
        }));
        setHistory(data);
      }
    }

    fetchData();
  }, [symbol, period]);

  useEffect(() => {
    function findMinMax(key) {
      const datas = history.map((node) => node[key]);
      return {
        min: Math.min(...datas),
        max: Math.max(...datas),
      };
    }
    function updateState() {
      const { min, max } = findMinMax("Close");
      setYRange([Math.round(min - 10), Math.round(max + 10)]);
    }
    updateState();
  }, [history]);

  const CustomizedRectangle = (props) => {
    const { formattedGraphicalItems } = props;
    // get first and second series in chart
    const highSeries = formattedGraphicalItems[0];
    const openSeries = formattedGraphicalItems[1];
    const closeSeries = formattedGraphicalItems[2];
    const lowSeries = formattedGraphicalItems[3];

    // render custom content using points from the graph
    return highSeries?.props?.points.map((highPoint, index) => {
      const openPoint = openSeries?.props?.points[index];
      const closePoint = closeSeries?.props?.points[index];
      const lowPoint = lowSeries?.props?.points[index];

      const diff = openPoint.y - closePoint.y;
      const hlDiff = highPoint.y - lowPoint.y;
      console.log(diff);
      return (
        <>
          <Rectangle
            key={highPoint.payload.Date}
            width={4}
            height={diff}
            x={openPoint.x - 4}
            y={closePoint.y}
            fill={diff < 0 ? "red" : diff > 0 ? "green" : "none"}
          />
          <Rectangle
            key={highPoint.payload.Date}
            width={1}
            height={hlDiff}
            x={highPoint.x - 2.3}
            y={lowPoint.y}
            fill={diff < 0 ? "red" : diff > 0 ? "green" : "none"}
          />
        </>
      );
    });
  };

  // Format the dates for display on the XAxis
  const formatXAxis = (tickItem) => {
    // Example: Format the date as 'MMM DD'
    return tickItem.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
  };

  return (
    <div className="h-screen w-screen bg-gray-900 flex flex-col overflow-hidden	text-xs">
      <div className="flex justify-between mx-auto w-screen p-3 border-b border-gray-700 	">
        <div className="pl-3 overscroll-none">
          <h3 className="font-semibold text-lg">Equity Lens</h3>
        </div>
        <div className="flex pr-3 gap-2 flex-row-reverse"></div>
      </div>

      <div className=" flex flex-row flex-1 w-screen mb-0 ">
        <div className="flex flex-col  w-full ">
          <div className="flex flex-row justify-between w-full  px-4 py-2 gap-2 border-b border-gray-700 ">
            <div className="flex gap-2">
              <select
                onChange={(e) => setSymbol(e.target.value)}
                className="border border-gray-700 bg-slate-900 px-1.5 py-1 rounded"
              >
                {symbols.map((s) => {
                  return (
                    <option key={s.index} value={s.Symbol}>
                      {s.Symbol}
                    </option>
                  );
                })}
              </select>
              <select
                onChange={(e) => setPeriod(e.target.value)}
                className="border border-gray-700 bg-slate-900 px-1.5 py-1 rounded"
              >
                {periods.map((p) => {
                  return (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  );
                })}
              </select>
            </div>

            <div className="flex flex-row gap-2">
              <label>
                <input
                  type="checkbox"
                  checked={isCandleStick}
                  onChange={(e) => setIsCandleStick(!isCandleStick)}
                />
                &nbsp;Candelstick
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={isLow}
                  onChange={(e) => setIsLow(!isLow)}
                />
                &nbsp;Low
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={isHigh}
                  onChange={(e) => setIsHigh(!isHigh)}
                />
                &nbsp;High
              </label>
            </div>
          </div>
          <ResponsiveContainer width="100%" className="h-full">
            <ComposedChart
              width="100%"
              data={history}
              margin={{
                top: 10,
                right: 30,
                left: 0,
                bottom: 10,
              }}
            >
              <CartesianGrid
                strokeDasharray="2 4"
                vertical={false}
                dominantBaseline={yRange}
                opacity={0.6}
              />
              <XAxis dataKey="Date" tickFormatter={formatXAxis} />
              <YAxis domain={yRange} axisLine={false}></YAxis>
              <Tooltip content={<CustomTooltip />} />
              {/* <Legend /> */}

              <Line
                type="monotone"
                dataKey="High"
                stroke="Green"
                strokeWidth={isHigh ? 2 : 0}
                dot={false}
                opacity={0.6}
              />
              <Line
                type="monotone"
                dataKey="Open"
                stroke="Yellow"
                strokeWidth={isHigh ? 2 : 0}
                dot={false}
                opacity={0.6}
              />
              <Line
                type="monotone"
                dataKey="Close"
                stroke="Blue"
                opacity={1}
                strokeWidth={isCandleStick ? 0 : 2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="Low"
                stroke="red"
                strokeWidth={isLow ? 2 : 0}
                opacity={0.6}
                dot={false}
              />

              {isCandleStick && <Customized component={CustomizedRectangle} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex-1 p-2 border-l hidden border-gray-700 invisible">
          <span>Stocks</span>
        </div>
      </div>
    </div>
  );
};

export default StockChart;
