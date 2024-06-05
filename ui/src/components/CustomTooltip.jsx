import React from "react";

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const formattedDate = new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(label);
    return (
      <div className="rounded-md bg-slate-600 flex flex-col p-2 text-sm">
        <p className="label">{formattedDate}</p>
        <p className="low text-red-500">{payload[0].value}</p>
        <p className="high text-green-500">{payload[1].value}</p>
        <p className="close text-blue-500">{payload[2].value}</p>
      </div>
    );
  }

  return null;
};

export default CustomTooltip;
