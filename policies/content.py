"""Canonical source content for the Solara Dynamics, Inc. HR policy corpus.

This module is the single source of truth for policy text. `generate_corpus.py`
renders each :class:`PolicyDoc` into the file format assigned to it
(Markdown, HTML, or PDF) under ``policies/``. Keeping the content here (rather
than hand-authoring three divergent file formats) guarantees that document
IDs, titles, effective dates, and section structure stay consistent across
formats and makes the corpus trivially regenerable.

All content describes a fictional company and fictional policies. No real
company, employee, or legal advice is represented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

COMPANY_NAME = "Solara Dynamics, Inc."
COMPANY_SHORT_NAME = "Solara Dynamics"
COMPANY_PRODUCT = "Solara Flow"
COMPANY_DOMAIN = "solaradynamics.example"
HR_TEAM_NAME = "People & Culture Team"
HRIS_NAME = "Compass HR"


@dataclass
class Section:
    heading: str
    paragraphs: List[str] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)


@dataclass
class PolicyDoc:
    doc_id: str
    title: str
    effective_date: str  # YYYY-MM-DD
    version: str
    owner: str
    fmt: str  # "md" | "html" | "pdf"
    summary: str
    sections: List[Section]


POLICIES: List[PolicyDoc] = [
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-PTO-001",
        title="Paid Time Off (PTO) Policy",
        effective_date="2024-01-01",
        version="2.1",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Describes how Solara Dynamics employees accrue, request, and use "
            "paid time off, including the Flexible PTO plan for Director-level "
            "and above employees."
        ),
        sections=[
            Section(
                heading="Purpose and Scope",
                paragraphs=[
                    f"{COMPANY_NAME} provides paid time off (PTO) so employees can rest, "
                    "handle personal matters, travel, and return to work refreshed and "
                    "productive. Time away from work is a normal and expected part of a "
                    "healthy working life, and this policy is designed to make it simple "
                    "to plan, request, and track. This policy applies to all U.S.-based "
                    "full-time and part-time employees on Solara Dynamics' direct "
                    "payroll. Contractors and employees engaged through an Employer of "
                    "Record (EOR) in another country follow the leave terms in their "
                    "contractor agreement or local EOR policy instead of this document, "
                    "because those arrangements are already governed by local statutory "
                    "leave requirements.",
                    "This policy should be read together with the Company Holiday "
                    "Schedule Policy (POL-HOL-001), which governs company-observed "
                    "holidays that do not count against PTO, and the Leave of Absence "
                    "Policy (POL-LEV-001), which governs longer or specialized leaves "
                    "such as parental or medical leave that are not part of routine PTO "
                    "usage.",
                ],
            ),
            Section(
                heading="Eligibility",
                paragraphs=[
                    "All active full-time employees are eligible for PTO starting on "
                    "their first day of employment, and accrual begins immediately even "
                    "though usage in the first 30 days is limited as described below. "
                    "Part-time employees scheduled for at least 20 hours per week are "
                    "eligible for prorated PTO based on scheduled hours; part-time "
                    "employees scheduled under 20 hours per week do not accrue PTO. "
                    "Interns on a fixed-term unpaid or stipend basis, and employees on an "
                    "approved unpaid personal leave of absence, do not accrue PTO during "
                    "that period, though any PTO accrued and unused before the leave "
                    "began remains available once the employee returns to active status.",
                    "New hires may request PTO as soon as it has accrued in Compass HR, "
                    "but usage within the first 30 calendar days of employment requires "
                    "manager approval and is discouraged except for pre-existing, "
                    "unavoidable commitments made before the employee accepted their "
                    "offer (for example, a wedding or a trip booked prior to the offer "
                    "date). Managers should apply this 30-day guidance flexibly and in "
                    "good faith rather than as a hard denial, since the intent is to "
                    "encourage new hires to build early momentum on their team, not to "
                    "penalize legitimate pre-existing plans.",
                    "Employees who change from part-time to full-time status, or vice "
                    "versa, have their accrual rate recalculated prospectively from the "
                    "effective date of the change; PTO already accrued under the prior "
                    "schedule is not reduced.",
                ],
            ),
            Section(
                heading="Accrual-Based PTO Plan",
                paragraphs=[
                    "Employees below the Director level participate in the "
                    "Accrual-Based PTO Plan. During years 1 through 3 of employment, "
                    "employees accrue 1.25 days (10 hours) of PTO per month, totaling 15 "
                    "days per year. Beginning in year 4 of continuous employment, the "
                    "accrual rate increases to 1.67 days (13.33 hours) per month, "
                    "totaling 20 days per year. The tenure milestone is calculated from "
                    "the employee's original hire date, and the increased rate applies "
                    "starting the first full month after the employee's third "
                    "anniversary. Accrual is prorated for part-time employees based on "
                    "their scheduled weekly hours relative to a standard 40-hour week; "
                    "for example, an employee scheduled for 24 hours per week accrues "
                    "60% of the standard monthly rate.",
                    "PTO accrues monthly and is credited to the employee's balance in "
                    "Compass HR on the last business day of each month, based on active "
                    "employment status during that month. Accrued PTO can be viewed at "
                    "any time in Compass HR under 'Time Off > My Balance,' which also "
                    "shows pending requests, a running year-to-date used total, and the "
                    "current carryover cap described in Section 7. Employees do not need "
                    "to request a balance report from the People & Culture Team; Compass "
                    "HR is the system of record for all PTO balances.",
                    "Example: an employee hired on March 15 in their second year of "
                    "employment accrues 10 hours per month. By the end of a full "
                    "calendar year (12 months of accrual), that employee will have "
                    "accrued 120 hours, equivalent to 15 eight-hour workdays, before "
                    "accounting for any PTO already used.",
                ],
            ),
            Section(
                heading="Flexible PTO Plan (Director and Above)",
                paragraphs=[
                    "Employees at the Director level and above participate in the "
                    "Flexible PTO Plan instead of the Accrual-Based PTO Plan. Flexible "
                    "PTO is not accrued and has no fixed annual maximum; time off is "
                    "taken as needed subject to manager approval and business needs. "
                    "Employees on the Flexible PTO Plan are strongly encouraged to take "
                    "a minimum of 15 days off per calendar year to support wellbeing and "
                    "avoid burnout, and managers are expected to actively encourage "
                    "direct reports to use this time rather than treat 'unlimited' as an "
                    "invitation to take less time off than the accrual-based plan would "
                    "otherwise provide.",
                    "Because Flexible PTO is not accrued, no PTO balance is tracked in "
                    "Compass HR for these employees, and no PTO payout is made upon "
                    "separation from the company, since there is no accrued liability to "
                    "pay out. Employees who are promoted into a Director-level role "
                    "transition from the Accrual-Based Plan to the Flexible Plan "
                    "effective the first day of the month following their promotion; any "
                    "PTO balance accrued up to that date is paid out at the time of "
                    "transition rather than carried forward indefinitely as an unused "
                    "balance under a plan that no longer tracks balances.",
                ],
            ),
            Section(
                heading="Requesting and Approving PTO",
                paragraphs=[
                    "All PTO requests must be submitted through Compass HR under 'Time "
                    "Off > Request Time Off' and approved by the employee's direct "
                    "manager before the time off is taken. Verbal or chat-message "
                    "approval alone does not satisfy this requirement; the request must "
                    "be recorded in Compass HR so that it is reflected in the team "
                    "calendar, payroll, and the employee's balance. Managers should "
                    "respond to PTO requests within 2 business days of submission.",
                    "Standard advance notice windows depend on the length of the "
                    "request:",
                ],
                bullets=[
                    "1-2 consecutive business days: submit at least 5 business days in advance.",
                    "3-5 consecutive business days: submit at least 10 business days in advance.",
                    "More than 5 consecutive business days: submit at least 30 calendar days in advance.",
                    "Emergency or unplanned PTO (illness, family emergency) may be approved retroactively at the manager's discretion, and employees should simply notify their manager as soon as reasonably possible rather than waiting to file paperwork first.",
                ],
            ),
            Section(
                heading="Minimum Notice and Blackout Periods",
                paragraphs=[
                    "Employees in the Finance department may not take PTO during the "
                    "last 5 business days of any fiscal quarter (the 'quarter-close "
                    "blackout') except for emergencies, due to financial close "
                    "obligations that require the full Finance team to be available. "
                    "Solara Dynamics' fiscal year follows the calendar year, so "
                    "quarter-close blackout periods fall in late March, late June, late "
                    "September, and late December of each year; exact dates are "
                    "published by the Finance team at the start of each quarter.",
                    "There is no company-wide blackout period outside of the "
                    "Finance-specific window described above; all other departments "
                    "follow only the standard notice windows in Section 5. Individual "
                    "managers may temporarily restrict PTO around product launches or "
                    "major customer commitments, but must communicate any such "
                    "restriction to their team at least 2 weeks in advance and should "
                    "avoid restricting previously-approved requests except in "
                    "extraordinary circumstances.",
                ],
            ),
            Section(
                heading="Carryover and Payout Rules",
                paragraphs=[
                    "Employees on the Accrual-Based PTO Plan may carry over up to 5 "
                    "unused PTO days into the next calendar year. Accrued PTO beyond the "
                    "5-day carryover cap is forfeited on December 31 unless applicable "
                    "state law requires otherwise. To avoid losing accrued time, "
                    "employees are encouraged to review their balance each November and "
                    "schedule any excess days before year-end.",
                    "Employees whose primary work location is California are not "
                    "subject to the forfeiture rule above: under California law, "
                    "accrued and unused PTO is treated as earned wages, carries over in "
                    "full without a cap, and is paid out in full upon separation, "
                    "regardless of the general policy described in this section. This "
                    "carve-out applies based on the employee's designated work location "
                    "in Compass HR, not their state of residence if different, and is a "
                    "clear example of how the Work Location, Interstate & International "
                    "Work Policy (POL-LOC-001) can change which specific rules apply to "
                    "an individual employee.",
                    "Upon voluntary or involuntary separation, employees on the "
                    "Accrual-Based PTO Plan are paid out for unused, accrued PTO up to "
                    "the plan's accrual cap described above (or in full for "
                    "California-based employees). Employees on the Flexible PTO Plan "
                    "receive no PTO payout upon separation, consistent with Section 4.",
                ],
            ),
            Section(
                heading="Interaction with Holidays and Leave",
                paragraphs=[
                    "PTO is not deducted for company-observed holidays listed in the "
                    "Company Holiday Schedule Policy (POL-HOL-001) or for approved "
                    "leaves of absence under the Leave of Absence Policy (POL-LEV-001). "
                    "If a scheduled PTO day falls on a company holiday, the holiday "
                    "takes precedence and the PTO day is credited back to the "
                    "employee's balance automatically by Compass HR the following "
                    "business day.",
                    "PTO taken immediately before or after an approved leave of absence "
                    "(for example, using a few PTO days before parental leave begins) is "
                    "permitted and is tracked separately in Compass HR so that it does "
                    "not extend the length of the protected leave itself.",
                ],
            ),
            Section(
                heading="Part-Time and Contractor PTO",
                paragraphs=[
                    "Part-time employees accrue PTO on a prorated basis calculated as "
                    "(scheduled weekly hours / 40) x the standard accrual rate for their "
                    "tenure tier, as described in Section 3. For example, a part-time "
                    "employee scheduled for 20 hours per week in their first year "
                    "accrues 0.5 x 10 hours = 5 hours of PTO per month.",
                    "Independent contractors and EOR-employed international team "
                    "members are not covered by this policy; their time-off terms are "
                    "defined in their individual contractor agreement or local EOR "
                    "employment contract, which may provide statutory leave "
                    "entitlements that differ significantly from the plans described "
                    "above. Questions about a contractor's or EOR employee's specific "
                    "time-off terms should be directed to the People & Culture Team "
                    "rather than assumed from this policy.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I take PTO during my first month? Usage is discouraged but "
                    "not prohibited; if you have a pre-existing, unavoidable "
                    "commitment made before your offer date, discuss it with your "
                    "manager, who may approve it even though you are within your "
                    "first 30 days.",
                    "What happens if my PTO request is denied? Managers may deny or "
                    "ask to reschedule a request if the timing conflicts with a "
                    "blackout period, insufficient notice, or a critical business "
                    "need, but should provide a reason and work with the employee to "
                    "find an alternative date.",
                    "Does a company holiday during my PTO count against my balance? "
                    "No. As described in Section 8, if an approved PTO day falls on a "
                    "company holiday listed in POL-HOL-001, that day is credited back "
                    "to your balance automatically.",
                    "I am a manager - can I approve my own PTO? No. All PTO requests, "
                    "including those from managers and executives, must be approved "
                    "by the requester's own manager (or, for the CEO, by the Board) "
                    "to keep the approval chain consistent across the company.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "Employees are responsible for submitting PTO requests in Compass "
                    "HR with sufficient notice, tracking their own balance, and "
                    "communicating planned absences to their immediate team. Managers "
                    "are responsible for reviewing and responding to requests "
                    "promptly, applying blackout and notice rules consistently across "
                    "their team, and ensuring adequate coverage during a direct "
                    "report's absence.",
                    "The People & Culture Team owns this policy, publishes any "
                    "updates to accrual rates or carryover rules, and is the point of "
                    "contact for disputes about balance calculations or exceptions "
                    "based on state law. Payroll is responsible for reflecting "
                    "approved PTO payouts accurately in an employee's final "
                    "paycheck upon separation.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-HOL-001",
        title="Company Holiday Schedule Policy",
        effective_date="2024-01-01",
        version="1.4",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Lists Solara Dynamics' observed U.S. holidays, floating holidays, "
            "the winter office closure, and holiday pay rules."
        ),
        sections=[
            Section(
                heading="Purpose",
                paragraphs=[
                    f"This policy defines the paid holidays observed by {COMPANY_NAME} "
                    "for U.S.-based employees each calendar year and explains how "
                    "holidays are handled for non-U.S. team members, non-exempt staff, "
                    "and employees whose PTO overlaps with a holiday. It is designed to "
                    "give every employee a predictable annual holiday calendar they can "
                    "plan around well in advance.",
                ],
            ),
            Section(
                heading="Observed U.S. Holidays",
                paragraphs=[
                    "Solara Dynamics observes the following 10 paid holidays each year "
                    "for all U.S.-based employees, regardless of work location, work "
                    "arrangement, or employment type (full-time or part-time):",
                ],
                bullets=[
                    "New Year's Day",
                    "Martin Luther King Jr. Day",
                    "Memorial Day",
                    "Juneteenth",
                    "Independence Day",
                    "Labor Day",
                    "Thanksgiving Day",
                    "Day after Thanksgiving",
                    "Christmas Eve",
                    "Christmas Day",
                ],
            ),
            Section(
                heading="Floating Holidays",
                paragraphs=[
                    "In addition to the 10 fixed holidays, each U.S.-based employee "
                    "receives 2 floating holidays per calendar year to use for holidays "
                    "of personal, cultural, or religious significance not on the "
                    "standard list above, such as Diwali, Lunar New Year, Eid, or Yom "
                    "Kippur. Floating holidays are prorated for employees hired after "
                    "July 1 of a given year (one floating holiday instead of two) and do "
                    "not carry over to the following year; unused floating holidays are "
                    "forfeited on December 31.",
                    "Floating holidays are requested through Compass HR in the same way "
                    "as PTO under the Paid Time Off Policy (POL-PTO-001), and the same "
                    "manager-approval requirement applies, although floating holidays "
                    "are tracked as a separate balance from standard PTO so that using "
                    "one does not reduce an employee's regular PTO balance.",
                ],
            ),
            Section(
                heading="Holidays for Non-U.S. Employees and Contractors",
                paragraphs=[
                    "Employees engaged through an Employer of Record outside the United "
                    "States observe the statutory public holiday schedule of their "
                    "country of employment, as documented in their local EOR employment "
                    "agreement, rather than the U.S. holiday list in Section 2. This "
                    "means, for example, that an employee working through Solara "
                    "Dynamics' EOR arrangement in the United Kingdom observes UK bank "
                    "holidays such as the Early May Bank Holiday instead of the U.S. "
                    "Independence Day holiday. Independent contractors do not receive "
                    "paid holidays under this policy, consistent with their engagement "
                    "as non-employees.",
                ],
            ),
            Section(
                heading="Winter Office Closure",
                paragraphs=[
                    "Solara Dynamics offices are closed from December 26 through "
                    "December 31 each year ('Winter Recess'). These days are paid "
                    "company closure days for all U.S.-based employees and are not "
                    "deducted from PTO balances or floating holiday balances. Winter "
                    "Recess is intended to give the whole company a shared period of "
                    "rest at year-end, and employees are discouraged from scheduling "
                    "meetings or expecting responses to non-urgent messages during this "
                    "window.",
                    "Non-exempt employees who are required to work during Winter Recess "
                    "to support customer operations (for example, a small on-call "
                    "customer support rotation) receive holiday premium pay as described "
                    "in Section 6, and are offered the choice of an alternate day off "
                    "within the following 60 days in addition to the premium pay.",
                ],
            ),
            Section(
                heading="Holiday Pay for Non-Exempt Employees",
                paragraphs=[
                    "Non-exempt (hourly) employees required to work on an observed "
                    "holiday or during Winter Recess receive 1.5x their regular hourly "
                    "rate for hours worked on that day, in addition to their standard "
                    "holiday pay for the day. Exempt (salaried) employees required to "
                    "work on a holiday do not receive additional pay, since their "
                    "compensation is not calculated on an hourly basis, but may take an "
                    "alternate day off within 30 days with manager approval to preserve "
                    "the intent of the holiday.",
                ],
            ),
            Section(
                heading="Holidays Falling on a Weekend",
                paragraphs=[
                    "When a fixed holiday falls on a Saturday, it is observed on the "
                    "preceding Friday. When a fixed holiday falls on a Sunday, it is "
                    "observed on the following Monday. The People & Culture Team "
                    "publishes the exact observed dates for the upcoming calendar year "
                    "in Compass HR no later than November 1 of the prior year, so "
                    "employees can plan travel and personal PTO around the full holiday "
                    "calendar well in advance.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "I am on an approved leave of absence when a holiday occurs - do "
                    "I still get holiday pay? Yes, holidays occurring during an "
                    "approved paid leave (such as parental leave) are paid as a "
                    "holiday rather than drawing down the leave balance, consistent "
                    "with the Leave of Absence Policy (POL-LEV-001).",
                    "Can I use a floating holiday and a fixed holiday on the same "
                    "day? No, floating holidays are for additional days beyond the "
                    "fixed list in Section 2 and cannot be stacked with a fixed "
                    "holiday that already falls on that date.",
                    "What if I am hired right as Winter Recess begins? New hires "
                    "starting during Winter Recess have their start date and pay "
                    "begin as normal; Winter Recess is a paid closure for active "
                    "employees, and the People & Culture Team will confirm the exact "
                    "arrangement with the new hire directly.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The People & Culture Team owns the annual holiday calendar, "
                    "publishes observed dates each November, and confirms local "
                    "public holiday schedules for EOR-employed international staff "
                    "each year. Managers are responsible for planning around Winter "
                    "Recess and holiday coverage needs in customer-facing functions "
                    "well in advance, and for approving any alternate-day requests "
                    "from exempt employees who worked a holiday.",
                    "Payroll is responsible for applying correct holiday premium pay "
                    "to non-exempt employees who work on an observed holiday or "
                    "during Winter Recess, and for ensuring floating holiday usage is "
                    "tracked separately from standard PTO in each pay period.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-RWK-001",
        title="Remote Work Policy",
        effective_date="2024-03-01",
        version="1.3",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Defines Solara Dynamics' onsite, hybrid, and remote work "
            "categories, collaboration expectations, and the rules for "
            "temporary 'Work From Anywhere' remote work."
        ),
        sections=[
            Section(
                heading="Purpose and Philosophy",
                paragraphs=[
                    f"{COMPANY_NAME} supports a flexible, outcomes-oriented work "
                    "environment because we believe employees do their best work when "
                    "they have control over where and, within reason, when they work. "
                    "This policy defines the work arrangement categories available to "
                    "employees and the expectations that come with each. It works "
                    "together with the Work Location, Interstate & International Work "
                    "Policy (POL-LOC-001), which governs where an employee may legally "
                    "and administratively perform work, and the Information Security & "
                    "Data Protection Policy (POL-SEC-001), which governs how employees "
                    "must connect to company systems while working outside an office.",
                ],
            ),
            Section(
                heading="Work Arrangement Categories",
                paragraphs=[
                    "Every employee is assigned one of three work arrangement "
                    "categories in Compass HR, recorded on their employee profile "
                    "alongside their designated work location:",
                ],
                bullets=[
                    "Onsite: works from an assigned Solara Dynamics office at least 4 days per week. This is typical for roles requiring hands-on lab access or frequent in-person customer visits.",
                    "Hybrid: works from an assigned Solara Dynamics office at least 2 days per week; the default arrangement for employees who live within commuting distance of the Austin, Denver, or Raleigh offices.",
                    "Remote: has no regular office attendance expectation and works from an approved home work location, whether or not that location is near a Solara Dynamics office.",
                ],
            ),
            Section(
                heading="Eligibility Criteria",
                paragraphs=[
                    "Remote and hybrid eligibility is determined by role, not by "
                    "tenure or performance rating. Roles that require regular access to "
                    "physical lab equipment, in-person customer demonstrations, or "
                    "office-based administrative duties (such as Facilities or certain "
                    "IT hardware roles) are designated Onsite or Hybrid in the job "
                    "requisition and offer letter. All other roles default to "
                    "Remote-eligible unless the hiring manager documents a specific "
                    "business reason otherwise and that reason is reviewed by the "
                    "People & Culture Team before the requisition is posted.",
                    "An employee's work arrangement category may change over time if "
                    "their role changes (for example, moving from an individual "
                    "contributor engineering role into a Facilities-coordination role), "
                    "but such a change is treated as a formal work arrangement change "
                    "under Section 8, not an automatic side effect of a role change.",
                ],
            ),
            Section(
                heading="Home Office Requirements and Stipend",
                paragraphs=[
                    "Remote and hybrid employees must maintain a dedicated, private "
                    "workspace with a reliable internet connection of at least 25 Mbps "
                    "download / 5 Mbps upload, and must connect to internal systems "
                    "only through SolaraSecure VPN per the Information Security & Data "
                    "Protection Policy (POL-SEC-001). A dedicated workspace does not "
                    "need to be a separate room, but it should be a consistent location "
                    "that supports confidential video calls and secure handling of any "
                    "company equipment.",
                    "New remote and hybrid employees receive a one-time $150 home "
                    "office setup stipend, separate from standard equipment issued "
                    "under the Company Equipment & Asset Policy (POL-EQP-001), which "
                    "can be used for items such as a desk, chair, or lighting. The "
                    "stipend is requested through SolaraExpense within the first 60 "
                    "days of employment or of transitioning to a remote or hybrid "
                    "arrangement.",
                ],
            ),
            Section(
                heading="Core Collaboration Hours",
                paragraphs=[
                    "All employees, regardless of work arrangement, must be reachable "
                    "and available for meetings between 10:00 a.m. and 3:00 p.m. in "
                    "their assigned local business time zone (Eastern, Central, "
                    "Mountain, or Pacific). Outside of core hours, employees may "
                    "structure their schedule flexibly to meet their role's "
                    "responsibilities, provided they remain responsive to reasonable "
                    "requests from teammates in other time zones within a business day.",
                    "Teams that span multiple time zones are encouraged to rotate "
                    "meeting times so that the same team members are not always the "
                    "ones asked to join outside their preferred working hours.",
                ],
            ),
            Section(
                heading="Temporary Remote Work ('Work From Anywhere')",
                paragraphs=[
                    "Employees may temporarily work from a location other than their "
                    "designated work location under the 'Work From Anywhere' "
                    "allowance, which is intended to support things like visiting "
                    "family, a change of scenery, or working from a vacation home for "
                    "a limited time, subject to the following limits:",
                ],
                bullets=[
                    "Up to 20 business days (approximately 4 weeks) per calendar year may be used without special approval, provided the destination is a U.S. state where Solara Dynamics is already payroll-registered (see POL-LOC-001) and within the continental U.S. time zones.",
                    "Requests exceeding 20 business days in a calendar year, requests to work from a U.S. state not yet payroll-registered, or any international destination, are not covered by this allowance and must instead go through the Work Location Change Request process defined in POL-LOC-001, regardless of the requested duration. This applies even when the total time requested, such as six consecutive weeks, might otherwise seem like a simple extension of the allowance.",
                    "Manager notice (via Compass HR) is required even for allowance-eligible temporary remote work of any length, so that the team calendar and any on-call schedules stay accurate.",
                ],
            ),
            Section(
                heading="Performance Expectations",
                paragraphs=[
                    "Work arrangement does not change performance expectations. "
                    "Remote and hybrid employees are evaluated on the same goals, "
                    "quality standards, and collaboration expectations as onsite "
                    "employees, using the same performance review cycle and rubric. "
                    "Managers may require temporary return-to-office arrangements for "
                    "specific business needs, such as an important customer visit or "
                    "team offsite, with at least 2 weeks' notice except in urgent "
                    "circumstances.",
                ],
            ),
            Section(
                heading="Requesting a Change in Work Arrangement",
                paragraphs=[
                    "Employees who wish to permanently change their work arrangement "
                    "category (for example, from Hybrid to Remote, or from Remote to "
                    "Hybrid) must submit a request through Compass HR with manager and "
                    "People & Culture approval. Approval considers role requirements, "
                    "team collaboration needs, and, if the request involves a change of "
                    "state or country, compliance with the Work Location, Interstate & "
                    "International Work Policy (POL-LOC-001). Decisions are typically "
                    "communicated within 10 business days of submission.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "I want to work from a different state for six weeks - does the "
                    "'Work From Anywhere' allowance cover that? Not automatically. "
                    "Six weeks is roughly 30 business days, which exceeds the 20 "
                    "business day no-approval allowance in Section 7, so the request "
                    "must go through the Work Location Change Request process in "
                    "POL-LOC-001, even if the destination state is one where Solara "
                    "Dynamics is already payroll-registered.",
                    "I want to work from another country for six weeks - is that "
                    "covered by this policy? International work of any length is "
                    "never covered by the 'Work From Anywhere' allowance in this "
                    "policy; it always requires review under POL-LOC-001, including "
                    "confirmation that the destination country is on the approved "
                    "Employer of Record list.",
                    "Do I need approval to work from a coffee shop across town for a "
                    "day? No, working from a different location within the same "
                    "metro area and state does not change your designated work "
                    "location or trigger any approval process; this policy is "
                    "concerned with work location changes that cross state or "
                    "country lines or that last for an extended period.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "Employees are responsible for notifying their manager of any "
                    "temporary remote work location and for submitting a Work "
                    "Location Change Request in Compass HR when a planned absence "
                    "exceeds the 'Work From Anywhere' allowance or involves a new "
                    "state or country, well before the relevant lead time in "
                    "POL-LOC-001. Managers are responsible for confirming team "
                    "coverage and collaboration expectations are still met during "
                    "any temporary remote work period.",
                    "The People & Culture Team maintains the eligibility criteria for "
                    "each work arrangement category and reviews permanent work "
                    "arrangement change requests. IT Security ensures that "
                    "SolaraSecure VPN access and MDM enrollment remain in place for "
                    "any employee working temporarily from a new location.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-LOC-001",
        title="Work Location, Interstate & International Work Policy",
        effective_date="2024-03-01",
        version="1.2",
        owner=HR_TEAM_NAME,
        fmt="html",
        summary=(
            "Explains why work location matters for payroll and legal "
            "compliance, lists Solara Dynamics' payroll-registered states and "
            "approved international EOR countries, and defines approval "
            "requirements for interstate and international work."
        ),
        sections=[
            Section(
                heading="Purpose",
                paragraphs=[
                    "Where an employee physically performs work determines which "
                    "state or country's tax withholding, employment law, and benefits "
                    "rules apply to that employee, independent of where the employee "
                    "was originally hired or where their manager or team is based. This "
                    "policy explains those requirements and the approval process for "
                    "working outside an employee's assigned work location, whether the "
                    "change is a short trip, an extended stay, or a permanent move.",
                ],
            ),
            Section(
                heading="Why Work Location Matters",
                paragraphs=[
                    "Solara Dynamics must withhold and remit state payroll taxes, "
                    "maintain workers' compensation coverage, and comply with "
                    "employment law in every state where an employee performs work, "
                    "even temporarily. Similar obligations exist at the country level "
                    "for international work, including tax residency rules and, for "
                    "the company itself, the risk of accidentally creating a taxable "
                    "'permanent establishment' in a country where Solara Dynamics has "
                    "no registered legal entity.",
                    "Working from an unregistered state or an unsupported country can "
                    "create unplanned tax filings, benefits complications, immigration "
                    "risk, and legal exposure for both the company and the employee. "
                    "For this reason, all interstate work beyond a short trip and all "
                    "international work must be approved in advance through the process "
                    "in Section 7, even when the Remote Work Policy's 'Work From "
                    "Anywhere' allowance would otherwise seem to cover the duration.",
                ],
            ),
            Section(
                heading="Payroll-Registered States",
                paragraphs=[
                    "Solara Dynamics is currently payroll-registered and able to "
                    "employ staff in the following 14 states: Texas, Colorado, North "
                    "Carolina, California, New York, Washington, Illinois, Georgia, "
                    "Florida, Massachusetts, Virginia, Arizona, Oregon, and Tennessee. "
                    "An employee's designated work location must be one of these "
                    "states unless a documented exception has been approved by the "
                    "People & Culture Team and Payroll. The current list of "
                    "payroll-registered states is also published in Compass HR under "
                    "'Company Resources > Payroll States' and is updated whenever a new "
                    "state registration is completed.",
                ],
            ),
            Section(
                heading="Temporary Interstate Work",
                paragraphs=[
                    "Temporary work from a state other than an employee's designated "
                    "work location is handled based on duration, as summarized below. "
                    "These thresholds exist because most states only require "
                    "additional withholding or registration once an employee has "
                    "worked there for a meaningful period, but the exact threshold "
                    "varies by state, so Solara Dynamics applies a single, simpler "
                    "internal standard across all states for consistency:",
                ],
                bullets=[
                    "Up to 30 consecutive calendar days in a state where Solara Dynamics is already payroll-registered: manager notice in Compass HR is sufficient; no additional approval is required.",
                    "31 to 90 consecutive calendar days in an already-registered state: requires advance approval from Payroll and the People & Culture Team at least 15 business days before the work begins, to review state tax withholding obligations and confirm no additional registration is triggered.",
                    "More than 90 consecutive calendar days, or any duration in a state where Solara Dynamics is not payroll-registered: treated as a permanent relocation request (see Section 6), not temporary work, regardless of the employee's stated intent to eventually return to their original work location.",
                ],
            ),
            Section(
                heading="Permanent Relocation to a New State",
                paragraphs=[
                    "Employees who wish to permanently relocate their designated work "
                    "location to a new U.S. state must submit a relocation request at "
                    "least 45 calendar days in advance through Compass HR. Relocation "
                    "to an already payroll-registered state requires approval from the "
                    "employee's VP and the People & Culture Team. Relocation to a state "
                    "where Solara Dynamics is not yet registered additionally requires "
                    "Payroll to confirm registration timelines, which typically take 30 "
                    "to 60 calendar days, and may temporarily affect benefits "
                    "enrollment continuity during the registration window.",
                    "Employees relocating permanently should also review the Paid Time "
                    "Off Policy (POL-PTO-001), since certain state-specific rules (such "
                    "as California's PTO payout requirements) apply based on the "
                    "employee's new designated work location once the relocation is "
                    "approved and effective.",
                ],
            ),
            Section(
                heading="International Remote Work",
                paragraphs=[
                    "International remote work is permitted only in countries where "
                    "Solara Dynamics maintains an active Employer of Record (EOR) "
                    "agreement. As of this policy's effective date, the approved "
                    "countries are the United Kingdom, Canada, Portugal, and Ireland. "
                    "Employees whose permanent work location remains in the United "
                    "States may request up to 6 consecutive weeks (30 business days) of "
                    "temporary international remote work per calendar year in an "
                    "approved country, sometimes referred to informally as a "
                    "'workation.' This 6-week allowance is measured in business days "
                    "and does not reset separately for each trip; it is a per-employee, "
                    "per-calendar-year total across all approved-country trips "
                    "combined.",
                ],
                bullets=[
                    "Requires Global Mobility review for tax residency and permanent-establishment risk before departure.",
                    "Requires a minimum of 20 business days' advance notice, submitted as a Work Location Change Request in Compass HR.",
                    "Requires the employee to complete and sign an International Remote Work Attestation confirming their travel dates and destination before departure.",
                    "Requires continued compliance with the Information Security & Data Protection Policy (POL-SEC-001), including restrictions on accessing Restricted-classification data from outside approved countries, and completion of any additional Data Handling Acknowledgment required for the employee's role.",
                ],
            ),
            Section(
                heading="Prohibited and Unsupported Locations",
                paragraphs=[
                    "Any country not on the approved EOR list in Section 6 is not "
                    "supported for remote work of any duration, including short trips "
                    "combined with business travel, due to unresolved tax, immigration, "
                    "and legal risk. This is true even if the employee holds "
                    "citizenship or a right to work in that country, because the "
                    "restriction is about Solara Dynamics' ability to lawfully employ "
                    "someone performing work there, not about the individual employee's "
                    "personal right to be in that country.",
                    "Exceptions require written approval from both the Chief Financial "
                    "Officer and General Counsel and are granted only in unusual "
                    "circumstances, such as a documented family emergency requiring a "
                    "brief period of work from a non-approved country while the "
                    "employee's primary residence and tax obligations remain "
                    "unaffected. Such exceptions are rare, are evaluated case by case, "
                    "and are never guaranteed.",
                ],
            ),
            Section(
                heading="Approval Process and Lead Time",
                paragraphs=[
                    "All interstate work beyond 30 days and all international remote "
                    "work requests are submitted through Compass HR as a 'Work Location "
                    "Change Request,' which routes automatically to the employee's "
                    "manager, Payroll, and the People & Culture Team (and additionally "
                    "to Global Mobility for international requests). Approval status "
                    "and any outstanding requirements are visible to the employee in "
                    "Compass HR throughout the review, and the requester receives an "
                    "email notification at each stage of approval or if additional "
                    "information is needed.",
                    "Employees are encouraged to submit a Work Location Change Request "
                    "as early as possible, even before all travel details are "
                    "finalized, since the review process (particularly Global Mobility "
                    "review for international requests) can take longer than the "
                    "minimum lead times listed above during particularly busy periods.",
                ],
            ),
            Section(
                heading="Tax and Immigration Disclaimer",
                paragraphs=[
                    "This policy describes Solara Dynamics' internal administrative "
                    "process and does not constitute individual tax, immigration, or "
                    "legal advice. Employees with questions about their personal tax "
                    "residency, visa status, or right to work in a given location "
                    "should consult a qualified professional in addition to following "
                    "this policy. Approval of a Work Location Change Request confirms "
                    "only that Solara Dynamics can support the arrangement "
                    "administratively; it does not guarantee a particular personal tax "
                    "outcome for the employee.",
                ],
            ),
            Section(
                heading="Worked Examples",
                paragraphs=[
                    "Example (interstate, approvable with review): An employee whose "
                    "designated work location is Austin, Texas asks to work from "
                    "Denver, Colorado for six consecutive weeks (about 30 business "
                    "days) to be near family. Colorado is a payroll-registered state "
                    "and the duration falls in the 31-90 day tier, so the request is "
                    "not automatically approved but can be approved with 15 business "
                    "days' notice once Payroll and the People & Culture Team confirm "
                    "there is no additional withholding issue.",
                    "Example (international, approvable with review): The same "
                    "employee instead asks to work from Lisbon, Portugal for six "
                    "consecutive weeks. Portugal is on the approved Employer of "
                    "Record list and six weeks is exactly at the international "
                    "allowance limit, so the request can be approved after Global "
                    "Mobility review, 20 business days' notice, and a signed "
                    "International Remote Work Attestation.",
                    "Example (not supported): The same employee asks to work from "
                    "Thailand for six consecutive weeks. Thailand is not on the "
                    "approved Employer of Record list, so the request is not "
                    "supported under this policy regardless of duration, unless the "
                    "CFO and General Counsel grant a rare written exception under "
                    "Section 7.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "Payroll owns the list of payroll-registered states and is "
                    "responsible for evaluating whether a temporary or permanent "
                    "location change triggers new withholding obligations. Global "
                    "Mobility owns the list of approved international EOR countries "
                    "and leads tax-residency and permanent-establishment review for "
                    "international requests. The People & Culture Team coordinates "
                    "the overall Work Location Change Request workflow and is the "
                    "primary point of contact for employees submitting a request.",
                    "Employees are responsible for submitting requests with accurate "
                    "dates and destinations and for not beginning travel or work from "
                    "a new location before their request is approved. Managers are "
                    "responsible for confirming the request supports business "
                    "continuity for the team before providing their approval.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-SEC-001",
        title="Information Security & Data Protection Policy",
        effective_date="2024-01-15",
        version="2.0",
        owner="Security & IT Team",
        fmt="md",
        summary=(
            "Defines data classification levels, acceptable use, MFA/VPN "
            "requirements, BYOD rules, and incident reporting obligations."
        ),
        sections=[
            Section(
                heading="Purpose and Scope",
                paragraphs=[
                    "This policy establishes minimum information security "
                    "requirements for all Solara Dynamics employees, contractors, and "
                    "interns who access company systems, applications, or data, "
                    "regardless of work location or work arrangement. It applies "
                    "equally to employees working from a company office, from home, or "
                    "temporarily from another approved location under the Remote Work "
                    "Policy (POL-RWK-001) or the Work Location, Interstate & "
                    "International Work Policy (POL-LOC-001).",
                ],
            ),
            Section(
                heading="Data Classification",
                paragraphs=[
                    "All company and customer data falls into one of four "
                    "classification levels, and every employee is responsible for "
                    "handling data according to its classification regardless of which "
                    "system it lives in:",
                ],
                bullets=[
                    "Public: approved for external release (marketing materials, published documentation, public job postings).",
                    "Internal: general business information not intended for external release (internal wikis, meeting notes, non-sensitive internal announcements).",
                    "Confidential: sensitive business information (financial forecasts, unreleased product plans, employee compensation and performance records).",
                    "Restricted: customer production data, authentication credentials, and personally identifiable information (PII), subject to the strictest handling controls described throughout this policy.",
                ],
            ),
            Section(
                heading="Acceptable Use of Company Systems",
                paragraphs=[
                    "Company-issued accounts and equipment are provided for business "
                    "use. Limited incidental personal use is permitted provided it does "
                    "not interfere with job performance, violate law, or expose company "
                    "systems to risk, for example installing unapproved software or "
                    "browser extensions with broad data access, or using a company "
                    "account to sign up for unrelated personal services.",
                    "Employees should have no expectation of privacy in their use of "
                    "company systems, which may be monitored and reviewed consistent "
                    "with applicable law and Solara Dynamics' internal monitoring "
                    "standards, primarily for security and operational purposes rather "
                    "than routine surveillance of individual employees.",
                ],
            ),
            Section(
                heading="Password and Multi-Factor Authentication Requirements",
                paragraphs=[
                    "All company accounts must use a unique password of at least 14 "
                    "characters managed through the company password manager, and "
                    "multi-factor authentication (MFA) is required on all systems that "
                    "support it, with no exceptions for convenience. Shared logins are "
                    "prohibited except for designated service accounts approved by IT "
                    "Security and documented in the service account registry.",
                    "Employees who lose access to their MFA device should contact the "
                    "Solara IT Helpdesk immediately for identity verification and "
                    "account recovery rather than asking a colleague to share "
                    "credentials or bypass MFA in any way.",
                ],
            ),
            Section(
                heading="Use of Personal Devices (BYOD)",
                paragraphs=[
                    "Personal devices may be used to access company email and chat "
                    "applications after enrolling in the company's mobile device "
                    "management (MDM) profile, which allows IT Security to enforce a "
                    "device passcode and remotely wipe company data if the device is "
                    "lost or the employee separates from the company. Personal devices "
                    "may not be used to access, store, or process Confidential or "
                    "Restricted data. All development, customer support, and "
                    "data-handling work must be performed on company-issued equipment.",
                ],
            ),
            Section(
                heading="Public Wi-Fi and VPN Requirements",
                paragraphs=[
                    "Employees working from any location other than a Solara Dynamics "
                    "office, including while traveling domestically or "
                    "internationally under an approved Work Location Change Request, "
                    "must connect to internal systems exclusively through SolaraSecure "
                    "VPN. Use of public or unsecured Wi-Fi networks without VPN "
                    "protection to access company systems is prohibited, whether from a "
                    "coffee shop, airport, hotel, or coworking space.",
                    "SolaraSecure VPN is available for all company-issued laptops and "
                    "should be connected before opening any internal application, email "
                    "client, or customer data system. Employees experiencing "
                    "persistent VPN connectivity issues while traveling should contact "
                    "the Solara IT Helpdesk rather than working around the VPN "
                    "requirement.",
                ],
            ),
            Section(
                heading="Customer Data Handling",
                paragraphs=[
                    "Restricted-classification customer data may only be accessed "
                    "using company-issued equipment, over SolaraSecure VPN, from a "
                    "location that complies with the Work Location, Interstate & "
                    "International Work Policy (POL-LOC-001). Restricted data may not "
                    "be exported to personal storage, personal email, or unapproved "
                    "third-party services under any circumstances, including for the "
                    "purpose of working around a technical limitation.",
                ],
            ),
            Section(
                heading="Incident Reporting",
                paragraphs=[
                    "Any suspected security incident, including lost or stolen "
                    "equipment, phishing attempts, or unauthorized account access, "
                    "must be reported to security@solaradynamics.example or the "
                    "Solara IT Helpdesk within 24 hours of discovery. Prompt reporting "
                    "is not subject to disciplinary action, even if the employee's own "
                    "action contributed to the incident, because early reporting "
                    "significantly reduces the potential impact of most security "
                    "incidents.",
                ],
            ),
            Section(
                heading="International Data Transfer Considerations",
                paragraphs=[
                    "Because data protection and export-control obligations vary by "
                    "country, Restricted-classification data may only be accessed from "
                    "the United States or from a country on the approved Employer of "
                    "Record list in the Work Location, Interstate & International Work "
                    "Policy (POL-LOC-001): United Kingdom, Canada, Portugal, and "
                    "Ireland. Employees approved for international remote work under "
                    "POL-LOC-001 must complete an additional Data Handling "
                    "Acknowledgment with IT Security prior to departure if their role "
                    "involves Restricted data, confirming they understand the "
                    "additional handling restrictions that apply while traveling.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I check work email on my personal phone? Yes, after "
                    "enrolling your device in the company's MDM profile as described "
                    "in Section 5, you may use a personal phone for email and chat, "
                    "but not for accessing Confidential or Restricted systems.",
                    "Is it safe to work from hotel Wi-Fi while traveling? Only if "
                    "you connect through SolaraSecure VPN first; hotel, airport, and "
                    "coffee shop networks are all considered public and unsecured "
                    "under this policy regardless of whether they require a "
                    "password.",
                    "What should I do if I accidentally click a phishing link? "
                    "Report it immediately to security@solaradynamics.example or the "
                    "Solara IT Helpdesk. Prompt reporting is never subject to "
                    "disciplinary action, and quick action significantly reduces "
                    "potential harm.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The Security & IT Team owns this policy, maintains the data "
                    "classification standard, administers SolaraSecure VPN and MDM "
                    "enrollment, and leads incident response for reported security "
                    "events. Every employee is responsible for following the "
                    "acceptable use, password, MFA, and VPN requirements in this "
                    "policy and for reporting suspected incidents within 24 hours.",
                    "Managers are responsible for ensuring their direct reports "
                    "complete Security Awareness Training on schedule and for "
                    "promptly notifying IT Security if a departing employee's access "
                    "needs to be revoked ahead of the standard offboarding process.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-EXP-001",
        title="Business Expense & Reimbursement Policy",
        effective_date="2024-01-01",
        version="1.6",
        owner="Finance Team",
        fmt="html",
        summary=(
            "Covers approved expense categories, per diem rates, submission "
            "deadlines, documentation requirements, and approval thresholds."
        ),
        sections=[
            Section(
                heading="Purpose",
                paragraphs=[
                    "This policy defines which business expenses Solara Dynamics "
                    "reimburses, how to submit them, and the approvals required, so "
                    "employees can be reimbursed quickly and Finance can maintain "
                    "accurate, auditable records of company spending.",
                ],
            ),
            Section(
                heading="Approved Expense Categories",
                paragraphs=[
                    "Reimbursable expense categories include business travel (see the "
                    "Business Travel Policy, POL-TRV-001), client meals and "
                    "entertainment, approved software and subscriptions under "
                    "$50/month not already provided by IT, professional development "
                    "(conferences, courses, certifications) up to $2,000 per employee "
                    "per year, and coworking space day-passes for remote employees "
                    "traveling for work who need a professional space for client calls.",
                    "Expenses outside these categories may still be reimbursed with "
                    "advance written approval from the employee's manager and Finance, "
                    "provided there is a clear, documented business purpose.",
                ],
            ),
            Section(
                heading="Per Diem Rates for Travel Meals",
                paragraphs=[
                    "Employees traveling for business may either submit itemized meal "
                    "receipts or claim a flat per diem instead: $75 per full travel day "
                    "for domestic travel, and $100 per full travel day for "
                    "international travel. Per diem amounts are prorated to $25 for "
                    "partial travel days (departure or return day). Employees should "
                    "choose one method (itemized or per diem) for an entire trip rather "
                    "than mixing methods within the same trip.",
                ],
            ),
            Section(
                heading="Submission Process and Deadlines",
                paragraphs=[
                    "All expenses must be submitted through the SolaraExpense tool "
                    "within 30 calendar days of the expense being incurred. Expenses "
                    "submitted more than 60 days after the expense date will not be "
                    "reimbursed except in cases of documented extenuating "
                    "circumstances approved by Finance, such as an extended medical "
                    "leave that delayed submission.",
                ],
            ),
            Section(
                heading="Receipts and Documentation Requirements",
                paragraphs=[
                    "Itemized receipts are required for any single expense over $25. "
                    "For expenses claimed under the meal per diem, no receipt is "
                    "required. All expense submissions must include a brief business "
                    "purpose and, for client-facing expenses, the names of attendees "
                    "and their company affiliation, so that Finance can properly "
                    "categorize the expense for accounting and tax purposes.",
                ],
            ),
            Section(
                heading="Corporate Card vs. Reimbursement",
                paragraphs=[
                    "Employees who travel for business more than once per quarter, or "
                    "whose role regularly incurs client entertainment expenses, are "
                    "issued a Solara Dynamics corporate card. Corporate card "
                    "transactions still require documentation in SolaraExpense within "
                    "30 days but do not require the employee to pay out of pocket. "
                    "All other employees are reimbursed for approved out-of-pocket "
                    "expenses via payroll within one pay cycle of approval, typically "
                    "within 10 business days.",
                ],
            ),
            Section(
                heading="Non-Reimbursable Expenses",
                paragraphs=[
                    "Solara Dynamics does not reimburse personal entertainment, "
                    "traffic or parking fines, personal travel upgrades beyond the "
                    "class defined in the Business Travel Policy (POL-TRV-001), alcohol "
                    "at internal (non-client) events without VP approval, or expenses "
                    "lacking a clear business purpose. Expense reports that include "
                    "non-reimbursable items will have those specific line items "
                    "rejected rather than the entire report being denied.",
                ],
            ),
            Section(
                heading="International Expenses and Currency Conversion",
                paragraphs=[
                    "Expenses incurred in a foreign currency are converted to U.S. "
                    "dollars using the exchange rate on the transaction date as "
                    "reported by the corporate card issuer, or, for cash "
                    "reimbursements, the rate published by SolaraExpense on the date "
                    "of submission. Employees should retain the original-currency "
                    "receipt in addition to the converted amount shown in "
                    "SolaraExpense.",
                ],
            ),
            Section(
                heading="Approval Thresholds",
                paragraphs=[
                    "Expense reports up to $1,000 require direct manager approval. "
                    "Expense reports between $1,000 and $5,000 require VP approval in "
                    "addition to manager approval. Expense reports over $5,000 "
                    "additionally require Finance approval before reimbursement is "
                    "processed. These thresholds apply to the total value of a single "
                    "expense report, not to each individual line item within it.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I expense a meal for a candidate I am interviewing? Yes, "
                    "candidate meals are treated the same as client meals under "
                    "Section 2 and should include the candidate's name and the role "
                    "they are interviewing for as the business purpose.",
                    "What if I lose a receipt for an expense over $25? Contact the "
                    "vendor for a duplicate receipt if possible; if that is not "
                    "possible, submit the expense with a written explanation and your "
                    "manager may still approve it at their discretion, though this "
                    "should not become a regular practice.",
                    "Can I be reimbursed in a currency other than U.S. dollars? No, "
                    "all reimbursements are paid in U.S. dollars through payroll "
                    "using the conversion approach in Section 8, regardless of the "
                    "currency the original expense was incurred in.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The Finance Team owns this policy, sets per diem rates and "
                    "approval thresholds, and audits expense reports for compliance. "
                    "Employees are responsible for submitting accurate, timely "
                    "expense reports with appropriate documentation and for "
                    "understanding which categories are reimbursable before "
                    "incurring an expense.",
                    "Managers are the first approver for their direct reports' "
                    "expense reports and are responsible for confirming a clear "
                    "business purpose before approving. VPs and Finance apply "
                    "additional review at the higher approval thresholds described "
                    "in Section 9.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-BEN-001",
        title="Employee Benefits Policy",
        effective_date="2024-01-01",
        version="1.5",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Summarizes medical/dental/vision plans, 401(k) matching, FSA, "
            "insurance, and benefits eligibility rules by employment type."
        ),
        sections=[
            Section(
                heading="Purpose and Eligibility",
                paragraphs=[
                    f"{COMPANY_NAME} offers a competitive benefits package to support "
                    "employee health and financial wellbeing. Full-time employees "
                    "(scheduled 30 or more hours per week) are eligible for medical, "
                    "dental, and vision coverage beginning the first of the month "
                    "following 30 days of continuous employment. Part-time employees "
                    "scheduled under 30 hours per week are not eligible for medical, "
                    "dental, or vision coverage but are eligible for the 401(k) plan "
                    "described in Section 4.",
                    "Independent contractors and EOR-employed international team "
                    "members are not eligible for benefits under this policy and "
                    "instead receive statutory benefits, if any, through their "
                    "contractor agreement or local EOR, which may include benefits "
                    "such as national health coverage that are not applicable to "
                    "U.S.-based direct employees.",
                ],
            ),
            Section(
                heading="Health, Dental, and Vision Plans",
                paragraphs=[
                    "Eligible employees may choose from three medical plan options: "
                    "Solara Bronze PPO, Solara Gold PPO, and Solara HDHP+HSA. The "
                    "Bronze PPO has the lowest monthly premium with higher out-of-pocket "
                    "costs; the Gold PPO has a higher premium with lower out-of-pocket "
                    "costs; the HDHP+HSA plan pairs a high-deductible health plan with a "
                    "Health Savings Account that Solara Dynamics contributes to "
                    "annually. Dental and vision coverage are offered as separate "
                    "elections and can be combined with any medical plan choice.",
                    "Solara Dynamics pays 80% of the employee-only premium for all "
                    "three medical plans and 60% of the premium for enrolled "
                    "dependents (spouse, domestic partner, and/or children). Detailed "
                    "plan summaries, including deductibles and out-of-pocket maximums, "
                    "are available in Compass HR under 'Benefits > Plan Documents.'",
                ],
            ),
            Section(
                heading="Waiting Period",
                paragraphs=[
                    "New employees must complete a 30-calendar-day waiting period "
                    "before medical, dental, and vision coverage begins. Coverage "
                    "becomes effective on the first day of the month following the "
                    "waiting period; for example, an employee starting on March 10 "
                    "completes the waiting period on April 9 and has coverage "
                    "effective May 1. Employees must complete benefits enrollment in "
                    "Compass HR within 14 days of their hire date or wait until the "
                    "next open enrollment period, unless they experience a qualifying "
                    "life event as described in Section 9.",
                ],
            ),
            Section(
                heading="401(k) Retirement Plan and Match",
                paragraphs=[
                    "All employees, including part-time employees, are eligible to "
                    "enroll in the 401(k) plan after 90 days of employment. Solara "
                    "Dynamics matches 100% of employee contributions on the first 3% "
                    "of eligible compensation and 50% of the next 2%, for a maximum "
                    "employer match of 4% of eligible compensation when an employee "
                    "contributes at least 5%. All employer matching contributions vest "
                    "immediately, meaning employees keep 100% of the match even if "
                    "they leave shortly after receiving it.",
                ],
            ),
            Section(
                heading="Flexible Spending Accounts",
                paragraphs=[
                    "Eligible employees may enroll in a Healthcare Flexible Spending "
                    "Account (FSA) and/or a Dependent Care FSA during open enrollment "
                    "or within 30 days of a qualifying life event. FSA contribution "
                    "limits follow the annual IRS limits published each year in "
                    "Compass HR, and unused Healthcare FSA funds up to the IRS-allowed "
                    "carryover amount roll over to the next plan year.",
                ],
            ),
            Section(
                heading="Life and Disability Insurance",
                paragraphs=[
                    "Solara Dynamics provides basic life insurance and short-term "
                    "disability insurance at no cost to all benefits-eligible "
                    "employees, at a coverage amount of one times annual base salary "
                    "for life insurance. Employees may purchase supplemental life "
                    "insurance for themselves and dependents at group rates during "
                    "enrollment, and long-term disability coverage is also available "
                    "as a voluntary, employee-paid benefit.",
                ],
            ),
            Section(
                heading="Wellness Stipend",
                paragraphs=[
                    "Benefits-eligible employees receive a $50 per month wellness "
                    "stipend, reimbursable through SolaraExpense, for expenses such as "
                    "gym memberships, fitness classes, or mental health app "
                    "subscriptions. Unused monthly stipend amounts do not roll over to "
                    "future months.",
                ],
            ),
            Section(
                heading="Benefits for Part-Time and Contract Workers",
                paragraphs=[
                    "As described in Section 1, part-time employees are eligible only "
                    "for the 401(k) plan and are not eligible for medical, dental, "
                    "vision, life, or disability insurance, or the wellness stipend. "
                    "Contractors and EOR-employed international staff are not "
                    "eligible for any benefit described in this policy, regardless of "
                    "how many hours they work.",
                ],
            ),
            Section(
                heading="Qualifying Life Events and Open Enrollment",
                paragraphs=[
                    "Open enrollment occurs each year in November for coverage "
                    "effective January 1. Outside of open enrollment, employees may "
                    "change benefits elections within 30 days of a qualifying life "
                    "event such as marriage, divorce, birth or adoption of a child, or "
                    "loss of other coverage, by submitting documentation in Compass "
                    "HR. Changes made outside a qualifying life event or open "
                    "enrollment window are not permitted under IRS Section 125 rules "
                    "governing pre-tax benefits elections.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "I am a part-time employee - can I buy into the medical plan at "
                    "full cost? No, medical, dental, and vision plans are only "
                    "available to full-time employees under this policy; part-time "
                    "employees cannot enroll even if willing to pay the full premium "
                    "themselves.",
                    "When does my 401(k) match start? The employer match begins as "
                    "soon as you enroll and start contributing, which can happen any "
                    "time after your 90-day eligibility date; there is no separate "
                    "waiting period for the match itself once you are contributing.",
                    "I just got married - how long do I have to add my spouse to my "
                    "plan? You have 30 days from the date of marriage to submit the "
                    "qualifying life event and required documentation in Compass HR; "
                    "after 30 days, you must wait until the next open enrollment.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The People & Culture Team owns benefits plan selection, runs "
                    "annual open enrollment, and is the point of contact for "
                    "eligibility questions and qualifying life event documentation. "
                    "Payroll administers premium deductions, FSA contributions, and "
                    "401(k) contribution processing each pay period.",
                    "Employees are responsible for completing enrollment within the "
                    "required windows, keeping dependent information current, and "
                    "reviewing plan documents before selecting a medical plan option. "
                    "Managers have no role in benefits elections and should direct "
                    "any benefits questions from their team to the People & Culture "
                    "Team.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-ONB-001",
        title="New Hire Onboarding Policy",
        effective_date="2024-01-01",
        version="1.3",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Describes the pre-start checklist, first-day/first-week "
            "milestones, the 30/60/90-day plan, and onboarding buddy program."
        ),
        sections=[
            Section(
                heading="Purpose",
                paragraphs=[
                    "This policy describes the standard onboarding process for all "
                    "new Solara Dynamics employees, ensuring every new hire has the "
                    "access, equipment, and support needed to be productive quickly "
                    "and to feel welcomed into the team from their very first day.",
                ],
            ),
            Section(
                heading="Pre-Start Checklist",
                paragraphs=[
                    "At least 5 business days before a new hire's start date, the "
                    "People & Culture Team completes account provisioning in Compass "
                    "HR, and IT ships equipment per the Company Equipment & Asset "
                    "Policy (POL-EQP-001). Managers are notified to prepare a 30/60/90 "
                    "day plan and assign an onboarding buddy before the start date, and "
                    "are reminded to schedule introductory meetings with key "
                    "collaborators during the new hire's first two weeks.",
                ],
            ),
            Section(
                heading="First Day Schedule",
                paragraphs=[
                    "New hires complete account activation, review key policies "
                    "including this document and the Workplace Conduct & "
                    "Anti-Harassment Policy (POL-CON-001), and meet with their manager "
                    "and onboarding buddy. Remote and hybrid new hires join a virtual "
                    "IT setup call to confirm VPN and equipment access, while onsite "
                    "new hires complete a short office orientation with a Facilities "
                    "team member.",
                ],
            ),
            Section(
                heading="First Week Milestones",
                paragraphs=[
                    "During the first week, new hires complete required compliance "
                    "training (see Section 7), meet key cross-functional partners, and "
                    "receive an overview of team goals and current projects from their "
                    "manager. New hires are also enrolled in the next available new-hire "
                    "orientation session hosted by the People & Culture Team, which "
                    "covers company history, values, and an overview of available "
                    "benefits.",
                ],
            ),
            Section(
                heading="30/60/90-Day Plan",
                paragraphs=[
                    "Every new hire's manager documents role-specific goals for the "
                    "first 30, 60, and 90 days in Compass HR. At each milestone, the "
                    "manager and new hire hold a check-in to review progress, gather "
                    "feedback on the onboarding experience, and adjust goals as needed "
                    "for the next phase.",
                ],
            ),
            Section(
                heading="Onboarding Buddy Program",
                paragraphs=[
                    "Each new hire is paired with an onboarding buddy: a peer outside "
                    "their direct reporting line who helps answer day-to-day questions "
                    "during the first 90 days, from 'who do I ask about expense "
                    "reports' to informal advice about team norms. Onboarding buddies "
                    "are recognized in the quarterly People & Culture Team newsletter "
                    "as a thank-you for their role in a new hire's success.",
                ],
            ),
            Section(
                heading="Required Compliance Training",
                paragraphs=[
                    "All new hires must complete Security Awareness Training, "
                    "Anti-Harassment Training, and Code of Conduct Training within 14 "
                    "calendar days of their start date. Managers are notified in "
                    "Compass HR if training is not completed on time, and repeated "
                    "reminders escalate to the People & Culture Team after 21 days.",
                ],
            ),
            Section(
                heading="Equipment and Access Provisioning",
                paragraphs=[
                    "Standard equipment and system access are provisioned according "
                    "to role as defined in the Company Equipment & Asset Policy "
                    "(POL-EQP-001). Access to Confidential or Restricted data systems "
                    "requires manager approval in addition to standard provisioning, "
                    "consistent with the Information Security & Data Protection "
                    "Policy (POL-SEC-001).",
                ],
            ),
            Section(
                heading="Onboarding for Remote New Hires",
                paragraphs=[
                    "Remote new hires receive equipment shipped to their home address "
                    "at least 3 business days before their start date and are invited "
                    "to a live virtual welcome session with the People & Culture Team "
                    "in addition to the standard first-day schedule, to help offset "
                    "the reduced in-person interaction of a fully remote start.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "My equipment did not arrive before my start date - what should "
                    "I do? Contact the Solara IT Helpdesk immediately; your manager "
                    "and onboarding buddy can help you get temporary access to "
                    "web-based tools while a replacement shipment is arranged.",
                    "When do I get access to Confidential systems? Standard access is "
                    "provisioned before your start date, but access to systems "
                    "holding Confidential or Restricted data requires a specific "
                    "manager approval, which is typically requested during your "
                    "first week once your exact responsibilities are confirmed.",
                    "Do I need to complete onboarding training before I can use "
                    "company systems? You can use standard systems from day one, but "
                    "you must complete Security Awareness, Anti-Harassment, and Code "
                    "of Conduct training within 14 days as described in Section 7.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The People & Culture Team owns the onboarding process end to "
                    "end, provisions Compass HR accounts, and hosts new-hire "
                    "orientation. The hiring manager is responsible for the 30/60/90 "
                    "day plan, assigning an onboarding buddy, and providing "
                    "day-to-day guidance during the new hire's first weeks.",
                    "IT is responsible for shipping and configuring equipment before "
                    "the start date and for running the first-day setup call for "
                    "remote and hybrid new hires. The new hire is responsible for "
                    "completing required training on time and raising questions "
                    "early rather than waiting until a 30/60/90 check-in.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-EQP-001",
        title="Company Equipment & Asset Policy",
        effective_date="2024-02-01",
        version="1.2",
        owner="IT Team",
        fmt="pdf",
        summary=(
            "Covers standard equipment by role, requesting additional gear, "
            "care and loss/theft handling, and return requirements."
        ),
        sections=[
            Section(
                heading="Purpose and Scope",
                paragraphs=[
                    "This policy governs the issuance, use, and return of physical "
                    "equipment and software licenses provided by Solara Dynamics to "
                    "employees and contractors, so that company assets are tracked "
                    "consistently and returned or replaced appropriately.",
                ],
            ),
            Section(
                heading="Standard Equipment Issued by Role",
                paragraphs=[
                    "All employees receive a company laptop (choice of MacBook Pro or "
                    "Lenovo ThinkPad), a laptop lock, and standard peripherals. "
                    "Remote and hybrid employees additionally receive an external "
                    "monitor, keyboard, and mouse, shipped to their home address. "
                    "Engineering and Data roles may request an upgraded laptop "
                    "configuration (additional memory or storage) through their "
                    "manager if their work requires it.",
                ],
            ),
            Section(
                heading="Requesting Additional Equipment",
                paragraphs=[
                    "Additional equipment beyond the standard issue (a second "
                    "monitor, headset, ergonomic accessories) can be requested through "
                    "the Solara IT Helpdesk with manager approval. Requests are "
                    "typically fulfilled within 5 business days for domestic "
                    "addresses; international shipments for approved EOR employees may "
                    "take longer due to customs processing.",
                ],
            ),
            Section(
                heading="Ownership and Care of Equipment",
                paragraphs=[
                    "All issued equipment remains the property of Solara Dynamics. "
                    "Employees are expected to take reasonable care of equipment, "
                    "keep operating systems and security software up to date when "
                    "prompted, and avoid installing unauthorized software that could "
                    "compromise device security, consistent with the Information "
                    "Security & Data Protection Policy (POL-SEC-001).",
                ],
            ),
            Section(
                heading="Loss, Theft, and Damage",
                paragraphs=[
                    "Lost, stolen, or damaged equipment must be reported to the "
                    "Solara IT Helpdesk and, if security-related, to IT Security per "
                    "the Information Security & Data Protection Policy (POL-SEC-001), "
                    "within 24 hours of discovery. Equipment lost or damaged due to "
                    "normal wear and tear is replaced at no cost to the employee; "
                    "equipment lost or damaged due to gross negligence may result in "
                    "the employee being charged the depreciated replacement cost.",
                ],
            ),
            Section(
                heading="Personal Use of Company Equipment",
                paragraphs=[
                    "Limited personal use of company equipment is permitted "
                    "consistent with the Acceptable Use section of the Information "
                    "Security & Data Protection Policy (POL-SEC-001). Company "
                    "equipment may not be used to run a personal business, mine "
                    "cryptocurrency, or be shared with household members.",
                ],
            ),
            Section(
                heading="Software and License Compliance",
                paragraphs=[
                    "Only software approved by IT may be installed on company "
                    "equipment. Employees must not install personal copies of "
                    "licensed software or use company software licenses for "
                    "non-business purposes. New software requests, including free "
                    "tools, are submitted through the Solara IT Helpdesk so that IT "
                    "Security can review them for compatibility and data-handling "
                    "risk.",
                ],
            ),
            Section(
                heading="Return of Equipment Upon Departure",
                paragraphs=[
                    "All company equipment must be returned within 10 business days "
                    "of an employee's last day of employment using the prepaid return "
                    "shipping label provided by IT. Equipment not returned within this "
                    "window may result in the depreciated value being deducted from "
                    "final pay to the extent permitted by applicable state law, and "
                    "the People & Culture Team will follow up directly with the "
                    "departing employee before any deduction is made.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I keep my laptop if I buy it at depreciated value? In some "
                    "cases yes; contact the Solara IT Helpdesk before your last day "
                    "to request a buyout quote, which must be approved by IT and "
                    "Finance before the standard return process is skipped.",
                    "What happens if I spill coffee on my keyboard? Report it to the "
                    "Solara IT Helpdesk right away; accidental damage from normal use "
                    "is replaced at no cost under Section 5, and prompt reporting "
                    "helps avoid further damage to the device.",
                    "Can I use my personal external monitor instead of the "
                    "company-issued one? Yes, personal peripherals are allowed, but "
                    "Solara Dynamics is not responsible for repair or replacement of "
                    "personally owned equipment.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The IT Team owns equipment provisioning, the approved software "
                    "catalog, and the return process for departing employees. "
                    "Employees are responsible for the reasonable care of issued "
                    "equipment, prompt reporting of loss, theft, or damage, and "
                    "returning equipment on time when they separate from the "
                    "company.",
                    "Managers are responsible for approving additional equipment "
                    "requests from their direct reports and for confirming that "
                    "departing team members understand the equipment return "
                    "requirement during offboarding conversations.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-LEV-001",
        title="Leave of Absence Policy",
        effective_date="2024-01-01",
        version="1.4",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Covers parental, medical/FMLA, bereavement, jury duty, military, "
            "and personal unpaid leave, including documentation and "
            "job-protection rules."
        ),
        sections=[
            Section(
                heading="Purpose and Scope",
                paragraphs=[
                    "This policy describes the types of leave of absence available to "
                    "eligible Solara Dynamics employees beyond standard PTO, including "
                    "parental, medical, bereavement, jury duty, military, and personal "
                    "leave, and explains how each interacts with job protection under "
                    "applicable law.",
                ],
            ),
            Section(
                heading="Parental Leave",
                paragraphs=[
                    "Eligible employees receive 12 weeks of paid parental leave for "
                    "the birth, adoption, or fostering of a child, regardless of "
                    "gender or parent role. Birth parents are additionally eligible "
                    "for up to 6 to 8 weeks of paid medical recovery leave through "
                    "short-term disability, for a combined total of up to "
                    "approximately 18 to 20 weeks. Parental leave must be taken within "
                    "12 months of the qualifying event and may be taken continuously "
                    "or, with manager approval, in blocks of no less than 2 weeks, for "
                    "example to ease a return to work.",
                ],
            ),
            Section(
                heading="Medical Leave and FMLA Coordination",
                paragraphs=[
                    "Employees with a serious health condition may take medical leave "
                    "supported by a healthcare provider's certification submitted to "
                    "the People & Culture Team. For employees who meet federal Family "
                    "and Medical Leave Act (FMLA) eligibility requirements (12 months "
                    "of employment, 1,250 hours worked in the prior 12 months, and "
                    "employment at a location with 50 or more employees within 75 "
                    "miles), medical leave runs concurrently with FMLA leave, "
                    "providing up to 12 weeks of job-protected leave per rolling "
                    "12-month period.",
                ],
            ),
            Section(
                heading="Bereavement Leave",
                paragraphs=[
                    "Employees receive up to 5 paid days of bereavement leave for the "
                    "death of an immediate family member (spouse, partner, child, "
                    "parent, or sibling) and up to 2 paid days for the death of an "
                    "extended family member. Additional unpaid bereavement leave may "
                    "be approved by the manager on a case-by-case basis.",
                ],
            ),
            Section(
                heading="Jury Duty and Civic Leave",
                paragraphs=[
                    "Solara Dynamics provides full pay for the duration of jury duty "
                    "service, with no limit on the number of days, upon presentation "
                    "of a jury summons to the People & Culture Team. Employees called "
                    "as a witness in a legal proceeding unrelated to Solara Dynamics "
                    "business receive unpaid civic leave with job protection.",
                ],
            ),
            Section(
                heading="Military Leave",
                paragraphs=[
                    "Employees who are members of the uniformed services are granted "
                    "leave and reemployment rights in accordance with the Uniformed "
                    "Services Employment and Reemployment Rights Act (USERRA) and "
                    "applicable state law, including continued benefits coverage "
                    "options during leave as required by law.",
                ],
            ),
            Section(
                heading="Personal Leave of Absence (Unpaid)",
                paragraphs=[
                    "Employees who have exhausted PTO and do not qualify for another "
                    "leave category may request an unpaid personal leave of absence "
                    "of up to 30 calendar days, subject to VP approval. Extensions "
                    "beyond 30 days are evaluated on a case-by-case basis and are not "
                    "guaranteed to preserve the employee's position, though the "
                    "company will make reasonable efforts to identify a suitable "
                    "returning role where possible.",
                ],
            ),
            Section(
                heading="Requesting a Leave and Required Documentation",
                paragraphs=[
                    "All leave requests are submitted through Compass HR under 'Leave "
                    "of Absence' at least 30 calendar days in advance when the need "
                    "for leave is foreseeable (such as parental leave), or as soon as "
                    "practicable for unforeseeable leave (such as a medical "
                    "emergency). Medical certifications must be submitted to the "
                    "People & Culture Team, not to the employee's manager, to protect "
                    "confidentiality; managers are informed only that leave has been "
                    "approved, not the underlying medical details.",
                ],
            ),
            Section(
                heading="Job Protection and Return to Work",
                paragraphs=[
                    "Employees on FMLA-qualifying medical leave or approved parental "
                    "leave are reinstated to the same or an equivalent position upon "
                    "return, consistent with applicable law. Employees on personal "
                    "leave of absence are not guaranteed reinstatement but will be "
                    "considered for open, suitable positions if their prior role is "
                    "no longer available at the time they are ready to return.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I use PTO to top up my pay during unpaid personal leave? "
                    "No, personal leave of absence under Section 7 is unpaid by "
                    "design; if you have unused PTO, you should use it before your "
                    "leave begins rather than during it.",
                    "Do I need to tell my manager the medical reason for my leave? "
                    "No, medical certifications go to the People & Culture Team only; "
                    "your manager is informed that leave was approved, not the "
                    "underlying medical details, as described in Section 8.",
                    "Can I take parental leave if I am adopting rather than giving "
                    "birth? Yes, the 12-week paid parental leave in Section 2 applies "
                    "equally to birth, adoption, and foster placement; only the "
                    "additional medical recovery leave is specific to birth parents.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The People & Culture Team owns leave administration, collects "
                    "and safeguards medical certifications, and determines FMLA "
                    "eligibility for each request. Employees are responsible for "
                    "submitting leave requests with as much advance notice as "
                    "possible and for providing required documentation promptly.",
                    "Managers are responsible for supporting a smooth handoff of "
                    "responsibilities before a leave begins and for welcoming the "
                    "employee back into their same or an equivalent role upon "
                    "return, without requesting medical details the employee has not "
                    "chosen to share.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-CON-001",
        title="Workplace Conduct & Anti-Harassment Policy",
        effective_date="2024-01-01",
        version="1.7",
        owner=HR_TEAM_NAME,
        fmt="md",
        summary=(
            "Sets standards of conduct, equal employment opportunity, "
            "anti-harassment/anti-discrimination rules, reporting, and "
            "non-retaliation protections."
        ),
        sections=[
            Section(
                heading="Purpose and Scope",
                paragraphs=[
                    f"{COMPANY_NAME} is committed to a respectful, inclusive, and safe "
                    "workplace for every employee, contractor, and visitor. This "
                    "policy applies to conduct in the workplace, at company events, "
                    "and in virtual or remote work settings, including messages sent "
                    "through company chat and video conferencing tools.",
                ],
            ),
            Section(
                heading="Standards of Conduct",
                paragraphs=[
                    "Employees are expected to treat colleagues, customers, and "
                    "partners with professionalism and respect, to act with honesty "
                    "and integrity, and to comply with all applicable laws and "
                    "company policies in the course of their work, both inside and "
                    "outside normal business hours when representing the company.",
                ],
            ),
            Section(
                heading="Equal Employment Opportunity",
                paragraphs=[
                    "Solara Dynamics is an equal opportunity employer and makes "
                    "employment decisions based on qualifications, merit, and "
                    "business need, without regard to race, color, religion, sex, "
                    "sexual orientation, gender identity, national origin, age, "
                    "disability, veteran status, or any other characteristic "
                    "protected by applicable law. This principle applies to hiring, "
                    "promotion, compensation, and every other aspect of employment.",
                ],
            ),
            Section(
                heading="Anti-Harassment and Anti-Discrimination",
                paragraphs=[
                    "Harassment or discrimination based on any protected "
                    "characteristic is strictly prohibited, whether committed by a "
                    "manager, coworker, customer, or vendor, and regardless of whether "
                    "it occurs in person, in writing, or through digital "
                    "communication tools. Prohibited conduct includes unwelcome "
                    "sexual advances, derogatory comments or jokes, and creation of a "
                    "hostile work environment, and applies equally at in-person "
                    "offices, virtual meetings, and offsite company events.",
                ],
            ),
            Section(
                heading="Reporting Procedures",
                paragraphs=[
                    "Employees who experience or witness conduct that may violate "
                    "this policy should report it to their manager, any People & "
                    "Culture Team member, or through the confidential Solara Ethics "
                    "Hotline, which accepts anonymous reports and is operated by an "
                    "independent third party. Reports may be made verbally or in "
                    "writing, and employees are never required to confront the "
                    "individual involved directly before reporting.",
                ],
            ),
            Section(
                heading="Investigation Process",
                paragraphs=[
                    "The People & Culture Team acknowledges all reports within 5 "
                    "business days and conducts a prompt, impartial investigation. "
                    "Investigations are kept as confidential as possible, consistent "
                    "with the need to conduct a thorough review and take appropriate "
                    "corrective action, and both the reporting employee and the "
                    "subject of the report are informed of the general outcome once "
                    "the investigation concludes.",
                ],
            ),
            Section(
                heading="Non-Retaliation",
                paragraphs=[
                    "Solara Dynamics strictly prohibits retaliation against any "
                    "employee who reports a concern in good faith or participates in "
                    "an investigation under this policy. Employees who believe they "
                    "have experienced retaliation should report it immediately using "
                    "the channels described in Section 5, and retaliation claims are "
                    "treated with the same seriousness as the original report.",
                ],
            ),
            Section(
                heading="Conflicts of Interest",
                paragraphs=[
                    "Employees must disclose actual or potential conflicts of "
                    "interest, including outside employment, financial interests in "
                    "competitors or vendors, and close personal relationships with "
                    "individuals they supervise, using the annual Conflict of "
                    "Interest disclosure in Compass HR, and should also disclose new "
                    "conflicts as they arise rather than waiting for the annual cycle.",
                ],
            ),
            Section(
                heading="Disciplinary Action",
                paragraphs=[
                    "Violations of this policy may result in disciplinary action up "
                    "to and including termination of employment, regardless of the "
                    "violator's position or tenure. The severity of disciplinary "
                    "action considers the nature and impact of the violation and any "
                    "prior conduct history, and is determined in consultation with "
                    "the People & Culture Team and, where appropriate, Legal.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I report a concern anonymously? Yes, the Solara Ethics "
                    "Hotline described in Section 5 accepts anonymous reports through "
                    "an independent third party; you do not have to give your name "
                    "to have a concern investigated.",
                    "What if the person I need to report is my own manager? Report "
                    "it to any People & Culture Team member or through the Solara "
                    "Ethics Hotline instead of your manager; you are never required "
                    "to report a concern to the person the concern is about.",
                    "Will I be punished for reporting something that turns out not "
                    "to be a violation? No, good-faith reports are protected under "
                    "the non-retaliation provisions in Section 6 even if the "
                    "investigation does not substantiate a policy violation.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The People & Culture Team owns this policy, operates the "
                    "reporting channels described in Section 5, and leads or "
                    "coordinates investigations. Every employee, manager, and "
                    "contractor is responsible for treating others with respect and "
                    "for reporting suspected violations rather than ignoring them.",
                    "Managers have an additional responsibility to model respectful "
                    "conduct, to escalate any concern raised by a direct report to "
                    "the People & Culture Team promptly, and to never attempt to "
                    "personally investigate or resolve a harassment complaint "
                    "themselves.",
                ],
            ),
        ],
    ),
    # ------------------------------------------------------------------
    PolicyDoc(
        doc_id="POL-TRV-001",
        title="Business Travel Policy",
        effective_date="2024-04-01",
        version="1.1",
        owner="Finance Team",
        fmt="pdf",
        summary=(
            "Covers booking tools, airfare/lodging guidelines, international "
            "travel approval, travel insurance, and duty-of-care support."
        ),
        sections=[
            Section(
                heading="Purpose and Scope",
                paragraphs=[
                    "This policy provides guidelines for booking and expensing "
                    "business travel in a way that balances employee comfort, cost "
                    "control, and duty of care for every employee traveling on "
                    "company business.",
                ],
            ),
            Section(
                heading="Booking Travel",
                paragraphs=[
                    "All business travel must be booked through the SolaraTravel "
                    "booking tool so that Solara Dynamics maintains visibility into "
                    "employee whereabouts for safety purposes, and so that negotiated "
                    "corporate rates are applied automatically. Travel booked outside "
                    "SolaraTravel is reimbursed only with prior Finance approval, and "
                    "such approval is granted sparingly.",
                ],
            ),
            Section(
                heading="Airfare, Lodging, and Ground Transportation",
                paragraphs=[
                    "Employees book economy class for domestic flights and flights "
                    "under 6 hours. Directors and above may book premium economy for "
                    "international flights over 6 hours. Lodging should be booked at "
                    "or below the SolaraTravel-recommended nightly rate for the "
                    "destination city, which is reviewed quarterly to reflect current "
                    "market rates. Ground transportation should favor ride-share or "
                    "public transit over rental cars where practical, though rental "
                    "cars are approved when they are the more cost-effective or "
                    "practical option for the trip.",
                ],
            ),
            Section(
                heading="International Travel Approval and Safety",
                paragraphs=[
                    "International business travel requires VP approval and at least "
                    "10 business days' advance notice to allow time for visa and "
                    "safety review, including a check against current government "
                    "travel advisories for the destination. Employees traveling "
                    "internationally are automatically enrolled in the company's "
                    "travel insurance and emergency assistance program described in "
                    "Section 5.",
                ],
            ),
            Section(
                heading="Travel Insurance",
                paragraphs=[
                    "All company-booked business travel, domestic and international, "
                    "includes travel insurance covering trip interruption, medical "
                    "emergencies, and emergency evacuation at no cost to the "
                    "employee. Coverage details, including the insurer's policy "
                    "number and 24-hour contact information, are available from the "
                    "People & Culture Team upon request and are also included in the "
                    "SolaraTravel booking confirmation email.",
                ],
            ),
            Section(
                heading="Combining Business Travel with Personal Time",
                paragraphs=[
                    "Employees may extend a business trip for personal time "
                    "('bleisure' travel) with manager approval, provided doing so "
                    "does not increase the cost of company-booked airfare compared to "
                    "the business-only itinerary. Personal-day expenses (lodging, "
                    "meals, transportation) during the extension are the employee's "
                    "responsibility and are not covered by travel insurance, which "
                    "applies only to the business portion of the trip.",
                ],
            ),
            Section(
                heading="Travel Advances",
                paragraphs=[
                    "Employees without a corporate card who require funds before a "
                    "trip may request a travel advance of up to $500 through Finance, "
                    "to be reconciled against actual expenses in SolaraExpense within "
                    "30 days of return. Unused advance funds must be returned to "
                    "Finance at the time of reconciliation.",
                ],
            ),
            Section(
                heading="Duty of Care and Emergency Contacts",
                paragraphs=[
                    "Employees traveling on company business have access to a 24/7 "
                    "emergency travel assistance line, provided at booking through "
                    "SolaraTravel, for medical, safety, or logistical emergencies "
                    "while traveling. Employees are encouraged to save this contact "
                    "information before departure rather than searching for it during "
                    "an actual emergency.",
                ],
            ),
            Section(
                heading="Frequently Asked Questions",
                paragraphs=[
                    "Can I keep airline miles or hotel points from business travel? "
                    "Yes, employees may keep personal loyalty program rewards earned "
                    "on business travel booked through SolaraTravel, provided doing "
                    "so does not increase the cost of the trip to the company.",
                    "Do I need approval for a short domestic trip to see a client? "
                    "Domestic travel does not require the VP approval required for "
                    "international travel in Section 4, but should still be booked "
                    "through SolaraTravel and follow your manager's normal approval "
                    "process for time away from other duties.",
                    "What if my flight is canceled while traveling internationally? "
                    "Call the 24/7 emergency travel assistance line described in "
                    "Section 8 first; the travel insurance coverage in Section 5 "
                    "also applies to trip interruptions such as cancellations.",
                ],
            ),
            Section(
                heading="Roles and Responsibilities",
                paragraphs=[
                    "The Finance Team owns this policy and the SolaraTravel booking "
                    "tool, and reviews travel advances and high-cost bookings. "
                    "Employees are responsible for booking through SolaraTravel, "
                    "following the class-of-service guidelines in Section 3, and "
                    "reconciling any travel advance within 30 days of return.",
                    "Managers approve travel requests for their direct reports and "
                    "confirm the trip's business purpose. VPs additionally approve "
                    "all international travel, consistent with the safety review "
                    "requirement in Section 4.",
                ],
            ),
        ],
    ),
]
