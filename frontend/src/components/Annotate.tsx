import { useEffect, useMemo, useRef, useState } from 'react';
import { cosmosAPI } from '../services/cosmosAPI';

// Types
type EventItem = { description: string; start_time: string; end_time: string; event: string };
type ClipInfo = { name: string; hasAnnotation: boolean };

function secondsToMMSS(sec: number) {
	const s = Math.max(0, Math.floor(sec));
	const mm = String(Math.floor(s / 60)).padStart(2, '0');
	const ss = String(s % 60).padStart(2, '0');
	return `${mm}:${ss}`;
}

export default function Annotate() {
	const [classes, setClasses] = useState<string[]>([]);
	const [currentClass, setCurrentClass] = useState<string>("");
	const [clips, setClips] = useState<ClipInfo[]>([]);
	const [selected, setSelected] = useState<string>("");
	const [videoUrl, setVideoUrl] = useState<string>("");
	const [events, setEvents] = useState<EventItem[]>([]);
	const [filterMissing, setFilterMissing] = useState(false);
	const [reloadKey, setReloadKey] = useState(0);
	const [playing, setPlaying] = useState(false);
	const videoRef = useRef<HTMLVideoElement>(null);
	const prevUrl = useRef<string>("");
	const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
	const [connectionStatus, setConnectionStatus] = useState<{ status: 'connecting' | 'success' | 'error'; message: string } | null>(null);
	const [loading, setLoading] = useState(false);

	function showToast(type: 'success' | 'error', message: string) {
		setToast({ type, message });
		window.setTimeout(() => setToast(null), 2500);
	}

	const AVAILABLE_EVENTS: string[] = useMemo(() => (
		[
			'Goal',
			'Goal Line Event',
			'Woodworks',
			'Shot on Target',
			'Red Card',
			'Yellow Card',
			'Penalty Shot',
			'Hat Trick'
		]
	), []);

	useEffect(() => {
		return () => { if (prevUrl.current) URL.revokeObjectURL(prevUrl.current); };
	}, []);

	const filteredClips = useMemo(() => filterMissing ? clips.filter(c => !c.hasAnnotation) : clips, [clips, filterMissing]);

	// Load classes on mount
	useEffect(() => {
		(async () => {
			try {
				setConnectionStatus({ status: 'connecting', message: 'Connecting to backend...' });
				const cls = await cosmosAPI.getClasses();
				if (cls.length > 0) {
					setClasses(cls);
					setCurrentClass(cls[0]);
					setConnectionStatus({ status: 'success', message: 'Successfully connected to backend' });
				} else {
					setConnectionStatus({ status: 'error', message: 'No classes found. Make sure 01_clips directory exists.' });
				}
			} catch (e) {
				console.error(e);
				const errorMessage = e instanceof Error ? e.message : 'Failed to connect to backend';
				setConnectionStatus({ status: 'error', message: errorMessage });
			}
		})();
	}, []);

	// Load clips when class changes
	useEffect(() => {
		if (!currentClass) return;
		(async () => {
			try {
				setLoading(true);
				const items = await cosmosAPI.getClips(currentClass);
				setClips(items);
				setSelected("");
				setEvents([]);
				if (prevUrl.current) { URL.revokeObjectURL(prevUrl.current); prevUrl.current = ""; setVideoUrl(""); }
			} catch (e) {
				console.error(e);
				showToast('error', 'Failed to load clips');
			} finally {
				setLoading(false);
			}
		})();
	}, [currentClass, reloadKey]);

	async function loadClip(name: string) {
		setSelected(name);
		// Set video URL from backend
		const url = cosmosAPI.getVideoUrl(currentClass, name);
		if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
		prevUrl.current = url; // Not an object URL, so no need to revoke
		setVideoUrl(url);
		
		// Load annotation
		try {
			const data = await cosmosAPI.getAnnotation(currentClass, name);
			setEvents(Array.isArray(data) ? data as EventItem[] : []);
		} catch {
			setEvents([]);
		}
	}

	function addQuickEvent(eventName: string) {
		if (!videoRef.current) return;
		videoRef.current.pause();
		setPlaying(false);
		const nowSec = Math.max(0, videoRef.current.currentTime || 0);
		const startSec = Math.max(0, nowSec - 3);
		const start = secondsToMMSS(startSec);
		const end = secondsToMMSS(nowSec);
		setEvents(prev => [...prev, { description: '', event: eventName, start_time: start, end_time: end }]);
	}

	function updateEvent(index: number, patch: Partial<EventItem>) {
		setEvents(prev => prev.map((e, i) => i === index ? { ...e, ...patch } : e));
	}

	function removeEvent(index: number) {
		setEvents(prev => prev.filter((_, i) => i !== index));
	}

	async function save() {
		try {
			if (!currentClass || !selected) {
				showToast('error', 'Select a clip to save annotation.');
				return;
			}
			const valid = events.every(e => e.description && e.event && e.start_time && e.end_time);
			if (!valid) {
				showToast('error', 'Complete all event fields before saving.');
				return;
			}
			await cosmosAPI.saveAnnotation(currentClass, selected, events);
			setClips(prev => prev.map(c => c.name === selected ? { ...c, hasAnnotation: true } : c));
			showToast('success', 'Annotation saved.');
		} catch (err) {
			console.error(err);
			showToast('error', 'Failed to save annotation.');
		}
	}

	return (
		<div className="max-w-[1600px] mx-auto p-6">
			{/* Connection Status */}
			{connectionStatus && (
				<div className={`bg-[#121212] border-2 rounded-lg shadow-lg p-4 mb-6 transition-all duration-300 ${
					connectionStatus.status === 'success' 
						? 'border-[#76B900] bg-[#11260a]' 
						: connectionStatus.status === 'error'
						? 'border-red-600 bg-[#2a0a0a]'
						: 'border-gray-600'
				}`}>
					<div className="flex items-center gap-3">
						{connectionStatus.status === 'success' && (
							<svg className="w-5 h-5 text-[#76B900]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
							</svg>
						)}
						{connectionStatus.status === 'error' && (
							<svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
							</svg>
						)}
						{connectionStatus.status === 'connecting' && (
							<svg className="w-5 h-5 text-gray-400 animate-spin" fill="none" viewBox="0 0 24 24">
								<circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
								<path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						)}
						<div className={`text-sm ${
							connectionStatus.status === 'success' ? 'text-[#76B900]' : connectionStatus.status === 'error' ? 'text-red-400' : 'text-gray-400'
						}`}>
							{connectionStatus.message}
						</div>
					</div>
				</div>
			)}

			<div className="grid grid-cols-1 lg:grid-cols-[320px,1fr] gap-6">
				{/* Sidebar */}
				<aside className="bg-[#121212] rounded-lg p-4 border-2 border-transparent hover:border-[#76B900] transition-all">
					<h3 className="text-[#76B900] font-semibold mb-3">Classes</h3>
					<select disabled={!classes.length || loading} value={currentClass} onChange={e => setCurrentClass(e.target.value)} className="w-full bg-[#1a1a1a] border border-gray-700 rounded px-3 py-2 text-sm mb-3">
						{classes.map(c => <option key={c} value={c}>{c}</option>)}
					</select>
					<label className="flex items-center gap-2 text-sm text-gray-300 mb-3">
						<input type="checkbox" checked={filterMissing} onChange={e => setFilterMissing(e.target.checked)} /> Show only missing
					</label>
					<div className="text-xs text-gray-400 mb-2">{clips.filter(c => c.hasAnnotation).length} / {clips.length} annotated</div>
					<div className="flex flex-col gap-2">
						{filteredClips.map(c => (
							<button key={c.name} onClick={() => loadClip(c.name)} className={`text-left px-3 py-2 rounded border ${selected === c.name ? 'bg-[#0f1b0a] border-[#76B900]' : 'bg-[#1a1a1a] border-gray-700'} hover:border-[#76B900] transition`}>{c.name}{c.hasAnnotation ? ' ✅' : ''}</button>
						))}
					</div>
				</aside>

				{/* Main */}
				<section className="space-y-4">
					<div className="bg-[#121212] rounded-lg p-4 border-2 border-transparent hover:border-[#76B900] transition">
						<div className="flex items-center gap-3">
							<div>Class: <b>{currentClass || '-'}</b></div>
							<div>Clip: <b>{selected || '-'}</b></div>
							<button onClick={() => setReloadKey(k => k + 1)} className="ml-auto px-3 py-2 bg-[#1a1a1a] border border-gray-700 rounded">Refresh</button>
							<button onClick={save} disabled={!selected || events.length === 0} className="px-3 py-2 bg-[#76B900] text-black rounded-md disabled:opacity-50">Save (S)</button>
						</div>
					</div>

					<div className="bg-[#121212] rounded-lg p-4 border-2 border-transparent hover:border-[#76B900] transition">
						{videoUrl ? (
							<video ref={videoRef} controls src={videoUrl} className="w-full h-[360px] bg-black rounded" />
						) : (
							<div className="h-[360px] border border-dashed border-gray-700 rounded grid place-items-center text-gray-500">Select a clip</div>
						)}
						<div className="flex gap-2 mt-3 text-sm">
							<button onClick={() => { if (!videoRef.current) return; if (videoRef.current.paused) { videoRef.current.play(); setPlaying(true); } else { videoRef.current.pause(); setPlaying(false); } }} className="px-3 py-1 bg-[#1a1a1a] border border-gray-700 rounded">{playing ? 'Pause' : 'Play'}</button>
						</div>
					</div>

					<div className="bg-[#121212] rounded-lg p-4 border-2 border-transparent hover:border-[#76B900] transition">
						<h3 className="text-[#76B900] font-semibold mb-3">Quick Event Cards</h3>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-3">
							{AVAILABLE_EVENTS.map(ev => (
								<button
									key={ev}
									onClick={() => addQuickEvent(ev)}
									className="text-left bg-[#1a1a1a] border border-gray-700 rounded-lg p-3 hover:border-[#76B900] hover:shadow-[0_0_12px_rgba(118,185,0,0.2)] transition"
								>
									<div className="text-sm font-medium text-gray-200">{ev}</div>
									<div className="text-xs text-gray-400">Add at current time</div>
								</button>
							))}
						</div>
					</div>

					<div className="bg-[#121212] rounded-lg p-6 border-2 border-transparent hover:border-[#76B900] transition-all duration-300">
						<div className="flex items-center justify-between mb-4">
							<h3 className="text-xl font-semibold text-[#76B900]">Events</h3>
							<span className="text-sm text-gray-400">{events.length} event{events.length !== 1 ? 's' : ''}</span>
						</div>
						{events.length === 0 && (
							<div className="text-center py-8 text-gray-400 text-sm border border-dashed border-gray-700 rounded-lg">
								No events yet. Use Quick Event Cards above or click "Add at current time" to create an event.
							</div>
						)}
						<div className="space-y-4">
							{events.map((ev, i) => {
								const mmssToSeconds = (mmss: string): number | null => {
									if (!mmss || mmss === '--:--') return null;
									const m = mmss.match(/^(\d{1,2}):(\d{2})$/);
									if (!m) return null;
									const minutes = parseInt(m[1], 10);
									const seconds = parseInt(m[2], 10);
									if (Number.isNaN(minutes) || Number.isNaN(seconds)) return null;
									return minutes * 60 + seconds;
								};

								const seekToEvent = (time: string) => {
									if (!videoRef.current) return;
									const seconds = mmssToSeconds(time);
									if (seconds == null) return;
									videoRef.current.currentTime = seconds;
									videoRef.current.play().catch(() => {});
									setPlaying(true);
								};

								return (
									<div 
										key={i} 
										className="group bg-[#0f0f0f] rounded-lg border border-gray-800 hover:border-[#76B900]/50 transition-all duration-300 hover:shadow-[0_0_12px_rgba(118,185,0,0.15)] overflow-hidden"
									>
										{/* Event Header */}
										<div className="p-4 border-b border-gray-800/50">
											<div className="flex items-start justify-between gap-4 mb-4">
												<div className="flex items-center gap-3 flex-wrap flex-1">
													<input
														value={ev.event}
														placeholder="Event type (Goal, Penalty Shot, etc.)"
														onChange={e => updateEvent(i, { event: e.target.value })}
														className="px-3 py-2 bg-gradient-to-r from-[#76B900]/20 to-[#87ca00]/20 border border-[#76B900]/30 text-[#76B900] placeholder:text-[#76B900]/50 font-semibold rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-[#76B900] focus:border-transparent min-w-[160px]"
													/>
													<button
														onClick={() => ev.start_time && seekToEvent(ev.start_time)}
														disabled={!ev.start_time || ev.start_time === '--:--'}
														className="px-3 py-2 bg-[#1a1a1a] border border-gray-700 hover:border-[#76B900] hover:bg-[#0f1b0a] text-[#76B900] text-xs font-medium rounded transition-all duration-200 flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
														title="Seek video to start time"
													>
														<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
														</svg>
														<input
															value={ev.start_time}
															placeholder="00:00"
															onChange={e => updateEvent(i, { start_time: e.target.value })}
															onClick={(e) => e.stopPropagation()}
															className="bg-transparent border-none outline-none text-[#76B900] w-12 text-xs font-medium placeholder:text-[#76B900]/50"
														/>
													</button>
													<div className="flex items-center gap-2">
														<span className="text-xs text-gray-500">→</span>
														<input
															value={ev.end_time}
															placeholder="00:00"
															onChange={e => updateEvent(i, { end_time: e.target.value })}
															className="px-2 py-2 bg-[#1a1a1a] border border-gray-700 rounded text-xs text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#76B900]/50 focus:border-transparent w-14 text-center"
														/>
													</div>
													<button
														onClick={() => removeEvent(i)}
														className="px-3 py-2 bg-red-900/20 border border-red-700/30 text-red-300 hover:bg-red-900/30 hover:border-red-700/50 text-xs font-medium rounded transition-all duration-200 flex items-center gap-1.5 ml-auto"
													>
														<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
														</svg>
														Delete
													</button>
												</div>
											</div>
											
											{/* Description - Prominent and Large */}
											<div className="mt-3">
												<label className="block text-xs font-medium text-[#76B900]/80 mb-2 uppercase tracking-wide">
													Description <span className="text-red-400">*</span>
												</label>
												<textarea
													value={ev.description}
													placeholder="Enter a detailed description of the event... (e.g., 'Player #10 scores a goal from outside the box, top right corner, assisted by Player #7')"
													onChange={e => updateEvent(i, { description: e.target.value })}
													rows={3}
													className="w-full bg-[#0a0a0a] border-2 border-gray-800 focus:border-[#76B900] text-gray-200 placeholder:text-gray-500 rounded-lg px-4 py-3 text-sm leading-relaxed focus:outline-none focus:ring-2 focus:ring-[#76B900]/30 transition-all duration-200 resize-none"
												/>
												<div className="mt-1 text-xs text-gray-500">
													{ev.description.length > 0 ? (
														<span className={ev.description.length > 100 ? 'text-yellow-400' : 'text-gray-500'}>
															{ev.description.length} characters {ev.description.length > 100 ? '(consider keeping under 100 chars)' : ''}
														</span>
													) : (
														<span className="text-red-400">Description is required</span>
													)}
												</div>
											</div>
										</div>
									</div>
								);
							})}
						</div>
					</div>
				</section>
			</div>
			{/* Toast */}
			{toast ? (
				<div className={`fixed bottom-6 left-1/2 -translate-x-1/2 px-4 py-2 rounded-md shadow-lg border text-sm z-50 ${toast!.type === 'success' ? 'bg-[#11260a] border-[#1f3a12] text-[#b6f398]' : 'bg-[#2a0a0a] border-[#471515] text-[#ffb3b3]'}`}>
					{toast!.message}
				</div>
			) : null}
		</div>
	);
}
