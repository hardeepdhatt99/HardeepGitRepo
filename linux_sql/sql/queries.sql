-- Group hosts by CPU number and sort by their memory size in descending order
SELECT
    info.cpu_number,
    usage.host_id,
    info.total_mem
FROM
    host_info info
    INNER JOIN host_usage usage
    ON info.id = usage.host_id
GROUP BY
    info.cpu_number, usage.host_id, info.total_mem
ORDER BY
    info.total_mem DESC;

-- Average used memory in percentage over 5 minute intervals for each host.
SELECT
    usage.host_id,
    info.hostname,
    DATE_TRUNC('hour', usage.timestamp) + DATE_PART('min', usage.timestamp)::INTEGER / 5 * interval '5 min' AS interv,
    AVG ((((info.total_mem - usage.memory_free)/info.total_mem::float) * 100)::INTEGER) AS avg_mem_used
FROM
    host_info info
    INNER JOIN host_usage usage
    ON info.id = usage.host_id
GROUP BY
    DATE_TRUNC('hour', usage.timestamp) + DATE_PART('min', usage.timestamp)::INTEGER / 5 * interval '5 min',
    usage.host_id,
    info.hostname
ORDER BY
    usage.host_id;
